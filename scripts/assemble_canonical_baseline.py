"""Preserve raw runs and assemble a fully traced baseline after official wording repair."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from alignlab.evaluation.canonical import assemble_canonical
from alignlab.evaluation.io import digest, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'repair', 'protocol', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    original, repair, protocol, out = map(Path, (args.original, args.repair, args.protocol, args.output))
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    first, second = load(original / 'generation_run.json'), load(repair / 'generation_run.json')
    if first['status'] != 'completed' or second['status'] != 'completed':
        raise ValueError('Both source generation runs must be completed')
    inputs = read_jsonl(protocol / 'ifeval.jsonl')
    changes = load(protocol / 'protocol.json')['corrections']
    rows = assemble_canonical(read_jsonl(original / 'generations.jsonl'),
                              read_jsonl(repair / 'generations.jsonl'), inputs,
                              [r['prompt_id'] for r in changes])
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / 'generations.jsonl', rows)
    status = {key: first[key] for key in ('model_path', 'model_revision', 'tokenizer_revision',
                                          'size_class', 'max_input_tokens', 'decoding')}
    status.update(status='completed', complete_evaluation=True, n_prompts=len(rows), generation_count=len(rows),
                  input_sha256=digest(inputs), output_path=str(out / 'generations.jsonl'),
                  assembly_not_a_single_contiguous_generation_run=True,
                  assembly_time_utc=datetime.now(timezone.utc).isoformat(),
                  assembly_code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  correction_ids=[r['prompt_id'] for r in changes],
                  source_runs=[{'directory': str(path), 'manifest_sha256': sha(path / 'generation_run.json'),
                                'generations_sha256': sha(path / 'generations.jsonl')} for path in (original, repair)],
                  source_elapsed_seconds_total=first['elapsed_seconds'] + second['elapsed_seconds'])
    (out / 'generation_run.json').write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
