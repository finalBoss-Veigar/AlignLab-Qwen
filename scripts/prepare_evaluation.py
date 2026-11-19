"""Freeze full IFEval and a seeded preference sample before model generation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from alignlab.evaluation.io import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--preference-count', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    source, out = Path(args.source), Path(args.output)
    if out.exists():
        raise FileExistsError('Use a new protocol directory; existing inputs are preserved')
    official = read_jsonl(source / 'ifeval.jsonl')
    official_inputs = read_jsonl(source / 'ifeval_input.jsonl')
    preferences = read_jsonl(source / 'test_prefs.jsonl')
    if len(official) != 541 or len(official_inputs) != 541 or {r['prompt'] for r in official} != {r['prompt'] for r in official_inputs}:
        raise ValueError('Expected complete 541-prompt IFEval source')
    if not 1 <= args.preference_count <= len(preferences):
        raise ValueError('Preference sample size must be nonempty and within the source pool')
    selected = random.Random(args.seed).sample(sorted(preferences, key=lambda r: str(r['prompt_id'])), args.preference_count)
    for rows in (official, selected):
        if len({r['prompt'] for r in rows}) != len(rows):
            raise ValueError('Duplicate prompt text; investigate before freezing protocol')
        if any(r.get('synthetic') is not False for r in rows):
            raise ValueError('Real-source metadata required')
    out.mkdir(parents=True)
    write_jsonl(out / 'ifeval.jsonl', official)
    write_jsonl(out / 'ifeval_input.jsonl', official_inputs)
    write_jsonl(out / 'preferences.jsonl', selected)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'status': 'inputs_frozen_not_generated', 'seed': args.seed,
                'counts': {'ifeval': len(official), 'preference_source': len(preferences), 'preferences': len(selected)},
                'sources': {p.name: sha(p) for p in source.glob('*.jsonl')},
                'outputs': {p.name: sha(p) for p in out.glob('*.jsonl')},
                'generation': {'do_sample': False, 'max_new_tokens': 2048, 'max_input_tokens': 4096,
                               'concurrency': 1, 'dtype': 'model_config_auto'},
                'comparison': ['original_instruct', 'continued_sft', 'sft_dpo'],
                'selection_note': 'Seeded prompt sample selected before any model output; never select by test score'}
    (out / 'protocol.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
