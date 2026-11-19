"""Recompute official scores and compare paired prompts without publishing answers."""
from collections import defaultdict
import random

from .bootstrap import percentile
from .generation import comparable_pairs, validate_generations
from .io import digest


def summarize_official(generations, scored):
    validate_generations(generations)
    generated = {r['prompt']: r for r in generations}
    if len(generated) != len(generations):
        raise ValueError('One generation per prompt is required')
    if len(scored) != len(generated) or len({r['prompt'] for r in scored}) != len(scored):
        raise ValueError('Official score coverage is incomplete or duplicated')
    records, groups = [], defaultdict(lambda: {'passed': 0, 'total': 0})
    for row in scored:
        source = generated.get(row['prompt'])
        if source is None or source['response'] != row['response']:
            raise ValueError('Scored answer differs from the saved generation')
        flags = row['follow_instruction_list']
        instructions = row['instruction_id_list']
        if (not flags or len(flags) != len(instructions) or
                any(type(x) is not bool for x in flags) or
                type(row['follow_all_instructions']) is not bool or
                row['follow_all_instructions'] != all(flags)):
            raise ValueError('Invalid official instruction flags')
        for instruction, passed in zip(instructions, flags):
            groups[instruction]['passed'] += passed
            groups[instruction]['total'] += 1
        records.append({'prompt_id': source['prompt_id'],
                        'prompt_sha256': digest(source['prompt']),
                        'response_sha256': digest(source['response']),
                        'passed': all(flags), 'instruction_ids': instructions,
                        'instruction_flags': flags,
                        'response_tokens': source.get('response_tokens'),
                        'output_limit_reached': source.get('response_tokens') == source['decoding']['max_new_tokens']})
    n = len(records)
    passed = sum(r['passed'] for r in records)
    instruction_n = sum(len(r['instruction_flags']) for r in records)
    instruction_passed = sum(sum(r['instruction_flags']) for r in records)
    return {'n_prompts': n, 'passed_prompts': passed, 'prompt_level': passed / n,
            'n_instructions': instruction_n, 'passed_instructions': instruction_passed,
            'instruction_level': instruction_passed / instruction_n,
            'output_limit_reached': sum(r['output_limit_reached'] for r in records),
            'by_instruction': dict(sorted(groups.items())), 'records': sorted(records, key=lambda r: r['prompt_id'])}


def paired_difference(base_rows, candidate_rows, base_scores, candidate_scores, *, seed=42, n_bootstrap=2000):
    if n_bootstrap < 1:
        raise ValueError('Positive bootstrap count required')
    pairs = comparable_pairs(base_rows, candidate_rows)
    base = summarize_official(base_rows, base_scores)
    candidate = summarize_official(candidate_rows, candidate_scores)
    lhs = {r['prompt_id']: r for r in base['records']}
    rhs = {r['prompt_id']: r for r in candidate['records']}
    differences, transitions = [], {'both_pass': 0, 'both_fail': 0, 'improved': 0, 'regressed': 0}
    for left, _ in pairs:
        a, b = lhs[left['prompt_id']], rhs[left['prompt_id']]
        if a['instruction_ids'] != b['instruction_ids']:
            raise ValueError('Official instructions differ across models')
        differences.append(int(b['passed']) - int(a['passed']))
        transitions['both_pass' if a['passed'] and b['passed'] else
                    'both_fail' if not a['passed'] and not b['passed'] else
                    'improved' if b['passed'] else 'regressed'] += 1
    rng = random.Random(seed)
    n = len(differences)
    bootstrap = [sum(rng.choice(differences) for _ in differences) / n for _ in range(n_bootstrap)]
    return {'definition': 'candidate minus baseline; one paired prompt is the resampling unit',
            'n_prompts': n, 'absolute_difference': sum(differences) / n,
            'percentage_point_difference': 100 * sum(differences) / n,
            'paired_bootstrap_95_ci': [percentile(bootstrap, .025), percentile(bootstrap, .975)],
            'seed': seed, 'n_bootstrap': n_bootstrap, 'transitions': transitions,
            'limitations': 'Single training seed; prompt bootstrap does not measure training-seed variability or remove benchmark contamination.'}
