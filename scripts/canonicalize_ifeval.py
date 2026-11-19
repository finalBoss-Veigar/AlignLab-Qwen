"""Freeze the exact pinned official IFEval wording and audit its one HF correction."""
import argparse
import hashlib
import json
from pathlib import Path
from alignlab.data.dedup import ngrams
from alignlab.data.schema import prompt_hash
from alignlab.evaluation.io import read_jsonl, write_jsonl
from alignlab.evaluation.ifeval import validate_inputs

REVISION = 'f8d6d14a1fc6eaef606d7d1e1fa4e96ee09afb81'
OFFICIAL_SHA = '67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-input', required=True)
    parser.add_argument('--old-protocol', required=True)
    parser.add_argument('--training-data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, old, out = Path(args.official_input), Path(args.old_protocol), Path(args.output)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(source) != OFFICIAL_SHA:
        raise ValueError('Official input differs from the verified pinned Git source')
    if out.exists():
        raise FileExistsError('Existing protocol preserved')
    official = read_jsonl(source)
    prior = read_jsonl(old / 'ifeval.jsonl')
    by_key = {r['key']: r for r in official}
    if len(official) != 541 or {r['key'] for r in prior} != set(by_key):
        raise ValueError('IFEval key coverage differs')
    corrected, changes = [], []
    for row in prior:
        standard = by_key[row['key']]
        item = dict(row)
        if row['prompt'] != standard['prompt']:
            changes.append({'key': row['key'], 'prompt_id': row['prompt_id'],
                            'before': row['prompt'], 'after': standard['prompt']})
        item.update(standard)
        item.update(hf_source_revision=row['revision'], hf_source_row_id=row['source_row_id'],
                    revision='git:' + REVISION, source_row_id=row['key'],
                    source_repository='https://github.com/google-research/google-research',
                    prompt_hash=prompt_hash(item['prompt']))
        corrected.append(item)
    if {r['key'] for r in changes} != {2785}:
        raise ValueError('Unexpected correction set; review before generating')
    standard = validate_inputs(official, [{'prompt': r['prompt'], 'response': ''} for r in official])
    fixed = validate_inputs(corrected, [{'prompt': r['prompt'], 'response': ''} for r in corrected])
    if sorted(standard, key=lambda r:r['key']) != sorted(fixed, key=lambda r:r['key']):
        raise ValueError('Canonical metadata still differs')
    query = ngrams(changes[0]['after'])
    isolation = {}
    for stage in ('sft', 'dpo'):
        for split in ('train', 'validation'):
            path = Path(args.training_data) / f'{stage}_{split}.json'
            rows = json.loads(path.read_text(encoding='utf-8'))
            maximum, count = 0.0, 0
            for row in rows:
                for message in row['messages']:
                    if message['role'] != 'user':
                        continue
                    grams = ngrams(message['content'])
                    score = len(query & grams) / len(query | grams)
                    maximum = max(maximum, score)
                    count += 1
                    if score >= 0.85:
                        raise ValueError(f'Canonical prompt leakage found in {path.name}')
            isolation[path.name] = {'sha256': sha(path), 'user_turns_checked': count, 'max_jaccard': maximum}
    out.mkdir(parents=True)
    write_jsonl(out / 'ifeval.jsonl', corrected)
    write_jsonl(out / 'ifeval_input.jsonl', official)
    write_jsonl(out / 'preferences.jsonl', read_jsonl(old / 'preferences.jsonl'))
    write_jsonl(out / 'ifeval_correction.jsonl', [r for r in corrected if r['key'] == 2785])
    protocol = json.loads((old / 'protocol.json').read_text(encoding='utf-8'))
    protocol.update(status='canonical_inputs_frozen_not_scored', official_revision=REVISION,
                    official_input_sha256=OFFICIAL_SHA, parent_protocol_sha256=sha(old / 'protocol.json'),
                    corrections=changes, additional_isolation=isolation,
                    outputs={p.name: sha(p) for p in out.glob('*.jsonl')})
    protocol['generation'].update(num_beams=1, min_new_tokens=0, repetition_penalty=1.05,
                                  eos_token_id=[151643,151645])
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': protocol['status'], 'correction_keys': [r['key'] for r in changes],
                      'canonical_prompts': len(corrected), 'additional_isolation': isolation}, indent=2))


if __name__ == '__main__':
    main()
