"""Descriptive paired diagnostics; never replace the full benchmark metric."""
from statistics import mean, median


def length_summary(values):
    if not values or any(type(v) is not int or v < 0 for v in values):
        raise ValueError('Expected nonempty nonnegative integer token lengths')
    ordered = sorted(values)
    # Nearest-rank p90; unlike an interpolated quantile this is an observed length.
    rank = (9 * len(ordered) + 9) // 10
    return {'count': len(values), 'mean': mean(values), 'median': median(values),
            'p90_nearest_rank': ordered[rank - 1], 'max': max(values)}


def paired_length_diagnostic(left, right):
    def index(rows):
        result = {}
        for row in rows:
            key = row['prompt_id']
            if key in result:
                raise ValueError('Duplicate prompt ID')
            if type(row['passed']) is not bool or type(row['output_limit_reached']) is not bool:
                raise ValueError('Expected boolean pass and length-cap flags')
            length_summary([row['response_tokens']])
            result[key] = row
        return result
    a, b = index(left), index(right)
    if not a or a.keys() != b.keys():
        raise ValueError('Paired prompt coverage must match')
    for key in a:
        if a[key]['prompt_sha256'] != b[key]['prompt_sha256'] or a[key]['instruction_ids'] != b[key]['instruction_ids']:
            raise ValueError('Paired prompt or official instructions differ')
    regressed = [k for k in a if a[k]['passed'] and not b[k]['passed']]
    uncapped = [k for k in a if not a[k]['output_limit_reached'] and not b[k]['output_limit_reached']]
    improved = [k for k in a if not a[k]['passed'] and b[k]['passed']]
    return {
        'n_paired': len(a), 'regressed': len(regressed), 'improved': len(improved),
        'regressed_with_candidate_at_cap': sum(b[k]['output_limit_reached'] for k in regressed),
        'regressed_with_candidate_below_cap': sum(not b[k]['output_limit_reached'] for k in regressed),
        'regressed_with_both_below_cap': sum(k in uncapped for k in regressed),
        'candidate_longer_count': sum(b[k]['response_tokens'] > a[k]['response_tokens'] for k in a),
        'candidate_shorter_count': sum(b[k]['response_tokens'] < a[k]['response_tokens'] for k in a),
        'left_lengths': length_summary([a[k]['response_tokens'] for k in a]),
        'right_lengths': length_summary([b[k]['response_tokens'] for k in a]),
        'both_below_cap_descriptive_subset': {
            'n': len(uncapped), 'left_passed': sum(a[k]['passed'] for k in uncapped),
            'right_passed': sum(b[k]['passed'] for k in uncapped),
            'difference': mean(int(b[k]['passed']) - int(a[k]['passed']) for k in uncapped) if uncapped else None,
        },
        'limitations': [
            'Post-hoc analysis of observed test outputs; no new model selection or full-score replacement.',
            'Below-cap selection depends on both model outputs and is not a causal length-controlled experiment.',
            'At-cap is an output length flag, not proof of truncation harm; below-cap does not exclude other length effects.',
            'Observed association cannot identify learning rate, data distribution or forgetting as the cause.',
        ],
    }
