"""Bounded AutoDL queue: real baseline, SFT, DPO, and one preregistered beta contrast.

Completed stages are validated and reused. Training resumes only from a complete
fingerprinted checkpoint. Incomplete generations are never scored or overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml
from alignlab.evaluation.io import digest, read_jsonl
from alignlab.training.runner import fingerprint, validate_resume, validate_training_data
from alignlab.training.queue_guard import acquire_queue_lock

REV = 'a09a35458c702b33eeacc393d103063234e8bc28'
ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / 'data/evaluation/protocol_20261119_canonical'
RESULTS = ROOT / 'runs/formal_20261119'
ARTIFACTS = ROOT / 'runs/artifacts/gpu_20261118/formal'
TRAIN = ROOT / '.venv-train/bin/python'
EVAL = ROOT / '.venv-eval/bin/python'
TOKENIZER = ROOT / 'models/tokenizer-qwen7b'
BASE = ROOT / 'models/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots' / REV


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--run-id', default='full_20261119')
    args = parser.parse_args()
    parsed_deadline = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
    if parsed_deadline.tzinfo is None:
        raise ValueError('Deadline requires an explicit timezone')
    deadline = parsed_deadline.timestamp()
    stages = [('base', None, BASE),
              ('sft', 'formal_sft', ARTIFACTS / 'sft-merged'),
              ('dpo', 'formal_dpo', ARTIFACTS / 'dpo-merged'),
              ('dpo_beta005', 'formal_dpo_beta005', ARTIFACTS / 'dpo-beta005-merged')]
    if not args.execute:
        print(json.dumps({'status': 'plan_only', 'deadline_utc': args.deadline_utc,
                          'stages': [s[0] for s in stages], 'input_protocol': load(INPUTS / 'protocol.json')}, indent=2))
        return
    if sys.platform != 'linux' or not Path('/root/autodl-tmp').is_dir():
        raise RuntimeError('This executor requires the authorized AutoDL Linux instance')
    if not args.run_id.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Simple run ID required')
    remaining = int(deadline - time.time())
    if not 0 < remaining <= 15 * 3600:
        raise ValueError('Deadline must be in the next 15 hours')
    queue_lock = acquire_queue_lock(ROOT)  # Before creating run directories or shutdown guards.
    os.chdir(ROOT)
    os.environ.update(HF_HOME=str(ROOT / 'models/hf-cache'), HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0', TOKENIZERS_PARALLELISM='false',
                      NLTK_DATA=str(ROOT / 'models/nltk_data'))
    run = ROOT / 'runs/queues' / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    state = {'status': 'running', 'queue_pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
             'deadline_utc': args.deadline_utc, 'stages': [], 'current': None,
             'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             'code_dirty': subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(),
             'protocol_sha256': file_hash(INPUTS / 'protocol.json')}

    def save():
        temporary = run / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        temporary.replace(run / 'status.json')

    # Separate guard still shuts the instance down if the queue process crashes.
    with (run / 'deadline-guard.log').open('w') as log:
        guard = subprocess.Popen(['/bin/bash', '-c', 'sleep "$1"; sync; /bin/bash /usr/bin/shutdown',
                                  'alignlab-deadline', str(remaining)], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state['deadline_guard_pid'] = guard.pid
    save()

    def execute(name, command):
        left = deadline - time.time()
        if left <= 0:
            raise TimeoutError('Authorized session deadline reached')
        state['current'] = name
        step = {'name': name, 'command': [str(c) for c in command],
                'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running'}
        state['stages'].append(step)
        save()
        print(f'START {name}', flush=True)
        with (run / (name + '.log')).open('w', encoding='utf-8') as log:
            child = subprocess.Popen(step['command'], cwd=ROOT, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            step['pid'] = child.pid
            save()
            try:
                code = child.wait(timeout=left)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                raise TimeoutError(f'Deadline during {name}')
        step.update(exit_code=code, status='completed' if code == 0 else 'failed',
                    ended_at=datetime.now(timezone.utc).isoformat())
        save()
        if code:
            raise RuntimeError(f'{name} exited {code}; see {run / (name + ".log")}')
        print(f'COMPLETE {name}', flush=True)

    def generation_command(model, revision, input_path, output):
        return [TRAIN, '-m', 'alignlab', 'generate', '--input', input_path, '--output', output,
                '--model', model, '--tokenizer', TOKENIZER, '--revision', revision,
                '--tokenizer-revision', REV, '--size-class', '7B', '--max-new-tokens', '2048',
                '--max-input-tokens', '4096', '--timeout-seconds', str(max(1, int(deadline-time.time()))),
                '--device', 'cuda:0', '--execute']

    try:
        for name, config_stem, model in stages:
            if name == 'dpo_beta005' and deadline - time.time() < 3 * 3600:
                state['stages'].append({'name': name, 'status': 'not_run', 'reason': 'less than 3h budget remains'})
                save()
                break
            if config_stem:
                config_path = ROOT / 'configs/gpu_20261118' / (config_stem + '.yaml')
                config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
                output = ROOT / config['output_dir']
                data = validate_training_data(ROOT / config['dataset_dir'], config)
                expected = fingerprint(config, data)
                completed = output / 'alignlab_training_manifest.json'
                if completed.exists():
                    proof = load(completed)
                    if proof.get('status') != 'completed' or proof.get('fingerprint') != expected:
                        raise ValueError(f'Training provenance mismatch: {name}')
                else:
                    command = [TRAIN, '-m', 'alignlab', 'train', config_path, '--execute']
                    checkpoints = sorted(output.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]), reverse=True)
                    if checkpoints:
                        validate_resume(checkpoints[0], expected)
                        command += ['--resume', checkpoints[0]]
                    execute(name + '_train', command)
                export_path = ROOT / 'configs/gpu_20261118' / (config_stem.replace('formal_', 'formal_export_', 1) + '.yaml')
                if not (model / 'alignlab_export_manifest.json').is_file():
                    execute(name + '_export', [TRAIN, '-m', 'alignlab', 'export', export_path, '--execute'])
            revision = REV if name == 'base' else file_hash(model / 'alignlab_export_manifest.json')
            original_base = RESULTS / 'base_ifeval'
            canonical_base = RESULTS / 'base_ifeval_canonical'
            if name == 'base' and (original_base / 'generation_run.json').exists() and not canonical_base.exists():
                state['current'] = 'base_ifeval_already_running'
                save()
                while load(original_base / 'generation_run.json')['status'] == 'running':
                    if time.time() >= deadline:
                        raise TimeoutError('Deadline while observing existing baseline')
                    time.sleep(10)
                old_status = load(original_base / 'generation_run.json')
                if old_status['status'] != 'completed':
                    raise RuntimeError('Original baseline failed; raw outputs preserved')
                correction = RESULTS / 'base_ifeval_repair'
                if not (correction / 'generation_run.json').exists():
                    execute('base_ifeval_repair', generation_command(model, revision,
                            INPUTS / 'ifeval_correction.jsonl', correction))
                execute('base_ifeval_assemble', [TRAIN, ROOT / 'scripts/assemble_canonical_baseline.py',
                    '--original', original_base, '--repair', correction, '--protocol', INPUTS, '--output', canonical_base])
            for label, source_file in [('ifeval', 'ifeval.jsonl'), ('preferences', 'preferences.jsonl')]:
                out = canonical_base if name == 'base' and label == 'ifeval' else RESULTS / (name + '_' + label)
                status_file = out / 'generation_run.json'
                if status_file.exists():
                    prior = load(status_file)
                    if (prior['status'] != 'completed' or prior['model_revision'] != revision or
                        prior['input_sha256'] != digest(read_jsonl(INPUTS / source_file)) or
                        prior['max_input_tokens'] != 4096 or prior['decoding'] != {
                            'do_sample': False, 'max_new_tokens': 2048, 'num_beams': 1,
                            'min_new_tokens': 0, 'repetition_penalty': 1.05,
                            'eos_token_id': [151643, 151645]}):
                        raise RuntimeError(f'Incomplete/incompatible generation preserved: {out}')
                else:
                    execute(name + '_' + label, generation_command(model, revision, INPUTS / source_file, out))
            score = RESULTS / (name + '_ifeval_score')
            responses = (canonical_base if name == 'base' else RESULTS / (name + '_ifeval')) / 'generations.jsonl'
            if (score / 'metrics.json').is_file():
                proof = load(score / 'run.json')
                if (proof.get('status') != 'completed' or proof.get('dataset_scope') != 'full_official_dataset' or
                    proof['input_sha256'] != digest(read_jsonl(INPUTS / 'ifeval_input.jsonl')) or
                    proof['response_sha256'] != digest(read_jsonl(responses))):
                    raise ValueError(f'Existing IFEval result is incompatible: {score}')
            else:
                execute(name + '_score', [EVAL, '-m', 'alignlab', 'ifeval', '--repo', ROOT / 'vendor/google-research',
                    '--input', INPUTS / 'ifeval_input.jsonl', '--responses', responses,
                    '--output', score, '--execute'])
        state.update(status='completed', current=None)
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        state['ended_at'] = datetime.now(timezone.utc).isoformat()
        save()
        # Allow result collection or operator repair; the independent deadline remains active.
        print(f"QUEUE {state['status']}; shutdown after at most 15 minutes for collection/repair", flush=True)
        time.sleep(max(0, min(900, deadline - time.time())))
        subprocess.run(['sync'], check=False)
        # AutoDL supplies a shell script without a shebang; direct exec raises ENOEXEC.
        # Calling Bash explicitly was verified on the actual instance after backups.
        subprocess.run(['/bin/bash', '/usr/bin/shutdown'], check=False)


if __name__ == '__main__':
    main()
