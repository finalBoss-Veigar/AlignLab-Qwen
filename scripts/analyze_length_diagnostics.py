"""Audit frozen train lengths and post-hoc paired failures without training a model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from alignlab.evaluation.diagnostics import length_summary, paired_length_diagnostic


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--evidence', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Keep previous analysis intact; choose a new output directory')
    output.mkdir(parents=True)
    started = time.perf_counter()
    run = {'experiment_id': output.name, 'started_at': datetime.now(timezone.utc).isoformat(),
           'status': 'running', 'purpose': 'post_hoc_length_diagnostic_no_new_training',
           'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
           'worktree_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True),
           'source_sha256': {f: sha(Path(f)) for f in ['scripts/analyze_length_diagnostics.py', 'src/alignlab/evaluation/diagnostics.py']},
           'command': [sys.executable, *sys.argv], 'python': platform.python_version(), 'platform': platform.platform(),
           'resource': 'local CPU only; saved token counts and official scores; no model inference',
           'random_seed': None, 'random_seed_reason': 'deterministic descriptive counts; original training seed remains 42'}
    try:
        data, evidence = Path(args.data), Path(args.evidence)
        manifest = read(data / 'data_manifest.json')
        result = {'input_sha256': {}, 'training_data': {}, 'paired_evaluation': {}}
        for task in ['sft', 'dpo']:
            file = data / (task + '_train.internal.jsonl')
            digest = sha(file)
            if digest != manifest['artifacts'][file.name]['sha256']:
                raise ValueError('Training token metadata no longer matches frozen manifest')
            result['input_sha256'][file.name] = digest
            rows = lines(file)
            if len(rows) != manifest['counts'][task]['sampled_train'] or any(x['split'] != 'train' or x['provenance']['synthetic'] for x in rows):
                raise ValueError('Expected complete real training split')
            fields = ['sequence_tokens', 'assistant_content_tokens'] if task == 'sft' else ['chosen_sequence_tokens', 'rejected_sequence_tokens', 'chosen_response_tokens', 'rejected_response_tokens']
            stats = {k: length_summary([r['token_lengths'][k] for r in rows]) for k in fields}
            if task == 'sft':
                stats['assistant_turns'] = length_summary([sum(m['role'] == 'assistant' for m in r['messages']) for r in rows])
                stats['length_definition'] = 'assistant_content_tokens sums assistant content across turns; not a single generated response length'
            else:
                deltas = [r['token_lengths']['chosen_response_tokens'] - r['token_lengths']['rejected_response_tokens'] for r in rows]
                if any(d != r['token_lengths']['response_length_delta'] for d, r in zip(deltas, rows)):
                    raise ValueError('Stored preference length delta is inconsistent')
                stats['chosen_longer'] = sum(d > 0 for d in deltas)
                stats['chosen_shorter'] = sum(d < 0 for d in deltas)
                stats['equal_length'] = sum(d == 0 for d in deltas)
                stats['interpretation'] = 'Training preference length association; cannot establish model length bias or human generation preference'
            result['training_data'][task] = stats
        records = {}
        for model in ['base', 'sft', 'dpo', 'dpo_beta005']:
            record = read(evidence / (model + '_evidence.json'))
            file = evidence / (model + '_strict_cases.jsonl')
            if sha(file) != record['recomputed']['strict']['cases_sha256']:
                raise ValueError('Case file differs from verified official evidence')
            records[model] = lines(file)
            if len(records[model]) != 541:
                raise ValueError('Expected all 541 official prompts')
            result['input_sha256'][file.name] = sha(file)
        for left, right in [('base', 'sft'), ('sft', 'dpo'), ('dpo', 'dpo_beta005')]:
            result['paired_evaluation'][right + '_minus_' + left] = paired_length_diagnostic(records[left], records[right])
        result['interpretation'] = 'Exploratory diagnosis of already observed results. No new benchmark score, human label, hyperparameter choice or causal claim.'
        (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        run['status'] = 'completed'
        run['result_sha256'] = sha(output / 'summary.json')
        print(json.dumps(result, indent=2))
    except BaseException as error:
        run['status'] = 'failed'; run['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        run['elapsed_seconds'] = time.perf_counter() - started
        run['ended_at'] = datetime.now(timezone.utc).isoformat()
        (output / 'run.json').write_text(json.dumps(run, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
