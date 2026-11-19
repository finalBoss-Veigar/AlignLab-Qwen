"""Validate saved formal training evidence and export an answer-free summary."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.evidence_root)
    queue = read(root / 'runs/queues/full_20261119_resume/status.json')
    assert queue['status'] == 'completed' and all(s['status'] == 'completed' and s['exit_code'] == 0 for s in queue['stages'])
    results = {'queue': {k: queue[k] for k in ('status', 'code_commit', 'code_dirty', 'protocol_sha256', 'started_at', 'ended_at')}, 'training': {}, 'stages': []}
    for s in queue['stages']:
        results['stages'].append({k: s[k] for k in ('name', 'started_at', 'ended_at', 'status', 'exit_code')} | {'wall_seconds': (datetime.fromisoformat(s['ended_at']) - datetime.fromisoformat(s['started_at'])).total_seconds()})
    configs = {}
    for name in ('sft', 'dpo', 'dpo-beta005'):
        folder = root / 'runs/artifacts/gpu_20261118/formal' / name
        manifest = read(folder / 'alignlab_training_manifest.json')
        exp = root / 'runs/experiments' / manifest['run_id']
        run = read(exp / 'run_manifest.json')
        audit = read(exp / 'audit/runtime_audit.json')
        state = read(folder / 'trainer_state.json')
        assert manifest['status'] == run['status'] == 'completed'
        assert run['code_commit'] == queue['code_commit'] and run['code_dirty'] == ''
        assert state['global_step'] == state['max_steps']
        assert audit['status'] == 'runtime_passed' and audit['prompt_system_padding_mask_verified']
        reference = audit.get('dpo_reference')
        if name != 'sft':
            assert all(reference[k] for k in ('initial_policy_reference_logps_equal', 'frozen_parameters_unchanged', 'probe_discarded_before_training'))
        merged = read(folder.parent / (name + '-merged') / 'config.json')
        assert merged['torch_dtype'] == 'bfloat16'
        configs[name] = manifest['config']
        files = [folder / f for f in ('alignlab_training_manifest.json', 'all_results.json', 'trainer_state.json')]
        files += [exp / 'run_manifest.json', exp / 'audit/runtime_audit.json', exp / 'telemetry.json']
        results['training'][name] = {
            'run_id': manifest['run_id'], 'status': manifest['status'], 'global_step': state['global_step'],
            'configured_max_steps': state['max_steps'], 'epoch_observed': state['epoch'],
            'config': manifest['config'], 'metrics': read(folder / 'all_results.json'),
            'telemetry': read(exp / 'telemetry.json'), 'export_dtype': merged['torch_dtype'],
            'audit_status': audit['status'], 'labels_mask_verified': True, 'dpo_reference': reference,
            'data_counts': run['data_revision']['counts'],
            'sha256': {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        }
    diff = {k: [configs['dpo'].get(k), configs['dpo-beta005'].get(k)] for k in set(configs['dpo']) | set(configs['dpo-beta005']) if configs['dpo'].get(k) != configs['dpo-beta005'].get(k)}
    assert set(diff) == {'pref_beta', 'output_dir'}
    assert results['training']['dpo']['dpo_reference']['frozen_sha256'] == results['training']['dpo-beta005']['dpo_reference']['frozen_sha256']
    results['beta_contrast_differences'] = diff
    results['limitations'] = ['SFT train_runtime belongs to the resumed segment; include the interrupted segment for end-to-end time/cost.', 'Peak VRAM is process allocated memory, not whole-device reservation or proof that a different GPU was tested.', 'Configured near-one epoch ends at framework max_steps; observed epochs are recorded, not rounded into complete sample coverage.', 'DPO rewards/accuracies is a chosen/rejected likelihood diagnostic, not human generation win rate.']
    target = Path(a.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'verified', 'runs': {k: v['global_step'] for k, v in results['training'].items()}, 'beta_changed_fields': sorted(diff)}))


if __name__ == '__main__':
    main()
