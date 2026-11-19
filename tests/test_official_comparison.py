"""Synthetic checks of evidence handling, never model performance results."""
from copy import deepcopy
import pytest
from alignlab.evaluation.comparison import summarize_official, paired_difference


def fixture():
    rows = [dict(prompt_id=str(i), prompt=f'prompt {i}', response=f'answer {i}',
                 model_id='fixture', model_revision='fixture-v1', tokenizer_revision='fixture-v1',
                 chat_template_sha256='fixture', source_split='fixture_test', synthetic=True,
                 size_class='synthetic', decoding={'do_sample': False, 'max_new_tokens': 10},
                 response_tokens=5) for i in range(4)]
    scores = [dict(prompt=r['prompt'], response=r['response'], instruction_id_list=['constraint'],
                   follow_instruction_list=[i < 2], follow_all_instructions=i < 2) for i, r in enumerate(rows)]
    return rows, scores


def test_recomputes_counts_and_rejects_changed_answer():
    rows, scores = fixture()
    assert summarize_official(rows, scores)['passed_prompts'] == 2
    scores[0]['response'] = 'changed'
    with pytest.raises(ValueError, match='answer differs'):
        summarize_official(rows, scores)


def test_rejects_missing_duplicate_and_inconsistent_flags():
    rows, scores = fixture()
    for invalid in [scores[:-1], scores[:3] + [scores[0]]]:
        with pytest.raises(ValueError, match='coverage'):
            summarize_official(rows, invalid)
    scores[0]['follow_all_instructions'] = False
    with pytest.raises(ValueError, match='flags'):
        summarize_official(rows, scores)


def test_paired_direction_and_zero_difference():
    rows, scores = fixture()
    same = paired_difference(rows, rows, scores, scores, n_bootstrap=100)
    assert same['paired_bootstrap_95_ci'] == [0, 0]
    better = deepcopy(scores)
    better[-1].update(follow_instruction_list=[True], follow_all_instructions=True)
    result = paired_difference(rows, rows, scores, better, n_bootstrap=100)
    assert result['percentage_point_difference'] == 25
    assert result['transitions']['improved'] == 1
    assert result['transitions']['regressed'] == 0


def test_refuses_different_decoding():
    rows, scores = fixture(); other = deepcopy(rows)
    for r in other:
        r['decoding']['max_new_tokens'] = 20
    with pytest.raises(ValueError, match='configuration mismatch'):
        paired_difference(rows, other, scores, scores, n_bootstrap=100)
