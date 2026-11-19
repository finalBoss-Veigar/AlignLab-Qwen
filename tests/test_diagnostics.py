import copy
import pytest
from alignlab.evaluation.diagnostics import paired_length_diagnostic, length_summary


def row(i, passed, capped=False, tokens=5):
    return {'prompt_id': str(i), 'prompt_sha256': str(i), 'passed': passed,
            'output_limit_reached': capped, 'response_tokens': tokens, 'instruction_ids': ['test']}


def test_cap_filter_keeps_pairs_and_preserves_full_regressions():
    left = [row(1, True), row(2, True), row(3, False), row(4, True, True)]
    right = [row(4, False), row(3, True), row(2, False), row(1, False, True)]
    got = paired_length_diagnostic(left, right)
    assert got['regressed'] == 3 and got['improved'] == 1
    assert got['regressed_with_candidate_at_cap'] == 1
    assert got['regressed_with_candidate_below_cap'] == 2
    assert got['regressed_with_both_below_cap'] == 1
    assert got['both_below_cap_descriptive_subset'] == {'n': 2, 'left_passed': 1, 'right_passed': 1, 'difference': 0}


def test_pairs_reject_missing_duplicate_or_changed_prompts():
    a = [row(1, True), row(2, False)]
    with pytest.raises(ValueError): paired_length_diagnostic(a, a[:1])
    with pytest.raises(ValueError): paired_length_diagnostic(a, [a[0], a[0]])
    b = copy.deepcopy(a); b[0]['prompt_sha256'] = 'different'
    with pytest.raises(ValueError): paired_length_diagnostic(a, b)


def test_all_capped_and_bad_lengths_do_not_create_invented_metrics():
    got = paired_length_diagnostic([row(1, True, True)], [row(1, False)])
    assert got['both_below_cap_descriptive_subset']['difference'] is None
    assert length_summary(list(range(1, 11)))['p90_nearest_rank'] == 9
    with pytest.raises(ValueError): length_summary([None])
    with pytest.raises(ValueError): length_summary([-1])
