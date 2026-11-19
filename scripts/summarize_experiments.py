"""Verify full official runs and export answer-free evidence for the repository."""
import argparse
import hashlib
import json
from pathlib import Path

from alignlab.evaluation.comparison import summarize_official, paired_difference
from alignlab.evaluation.io import read_jsonl, digest, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.results), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = {'models': {}, 'comparisons': {}, 'pending': [], 'human_preference': None,
               'human_preference_reason': 'No human labels supplied to this official-IFEval summarizer'}
    complete = {}
    for model in ('base', 'sft', 'dpo', 'dpo_beta005'):
        generation = root / (model + '_ifeval' + ('_canonical' if model == 'base' else ''))
        score = root / (model + '_ifeval_score')
        if not (score / 'run.json').is_file():
            summary['pending'].append(model)
            continue
        state = json.loads((score / 'run.json').read_text())
        gs = json.loads((generation / 'generation_run.json').read_text())
        rows = read_jsonl(generation / 'generations.jsonl')
        inputs = read_jsonl(score / 'official_inputs.jsonl')
        if (state['status'] != 'completed' or gs['status'] != 'completed' or
                not gs['complete_evaluation'] or len(rows) != 541 or
                state['dataset_scope'] != 'full_official_dataset' or
                state['response_sha256'] != digest(rows) or state['input_sha256'] != digest(inputs) or
                any(r['synthetic'] for r in rows)):
            raise ValueError(f'{model}: incomplete or untraceable official evaluation')
        evidence = {'generation': gs, 'scoring': state, 'recomputed': {}, 'files_sha256': {}}
        for mode in ('strict', 'loose'):
            scores = read_jsonl(score / ('eval_results_' + mode + '.jsonl'))
            recomputed = summarize_official(rows, scores)
            for metric in ('prompt_level', 'instruction_level', 'n_prompts', 'n_instructions'):
                if recomputed[metric] != state['metrics'][mode][metric]:
                    raise ValueError(f'{model}: reported {mode}/{metric} does not match raw scores')
            cases = out / (model + '_' + mode + '_cases.jsonl')
            write_jsonl(cases, recomputed.pop('records'))
            recomputed['cases_file'] = cases.name
            recomputed['cases_sha256'] = hashlib.sha256(cases.read_bytes()).hexdigest()
            evidence['recomputed'][mode] = recomputed
            if mode == 'strict':
                complete[model] = (rows, scores)
        for file in [generation / 'generations.jsonl', generation / 'generation_run.json',
                     score / 'eval_results_strict.jsonl', score / 'eval_results_loose.jsonl', score / 'run.json']:
            evidence['files_sha256'][str(file.relative_to(root))] = hashlib.sha256(file.read_bytes()).hexdigest()
        write_json(out / (model + '_evidence.json'), evidence)
        summary['models'][model] = {mode: {k: v for k, v in value.items() if k not in ('records', 'by_instruction', 'cases_file', 'cases_sha256')}
                                   for mode, value in evidence['recomputed'].items()}
    for baseline, candidate in [('base', 'sft'), ('sft', 'dpo'), ('base', 'dpo'), ('dpo', 'dpo_beta005')]:
        if baseline in complete and candidate in complete:
            a, sa = complete[baseline]; b, sb = complete[candidate]
            summary['comparisons'][candidate + '_minus_' + baseline] = paired_difference(a, b, sa, sb)
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
