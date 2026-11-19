"""Synthetic records check provenance logic, never report model performance."""
from copy import deepcopy
import pytest
from alignlab.evaluation.canonical import assemble_canonical
from alignlab.evaluation.io import digest


def row(key, prompt, answer):
    return dict(prompt_id=key, prompt=prompt, response=answer, model_id='test-model', model_revision='test',
                tokenizer_revision='test', chat_template_sha256='test', source_split='ifeval', synthetic=True,
                size_class='synthetic', decoding={'do_sample':False, 'max_new_tokens':20},
                context_sha256=digest([{'role':'user','content':prompt}]))


def fixture():
    return [row('a','unchanged','keep'),row('b','old wording','old answer')], [row('b','new wording','new answer')], [
        {'prompt_id':'a','prompt':'unchanged'}, {'prompt_id':'b','prompt':'new wording'}]


def test_only_new_generation_can_answer_corrected_prompt():
    old, new, prompts = fixture()
    untouched = deepcopy(old)
    output = assemble_canonical(old, new, prompts, ['b'])
    assert [r['response'] for r in output] == ['keep', 'new answer']
    assert output[0]['source_generation_record_sha256'] == digest(old[0])
    assert old == untouched


@pytest.mark.parametrize('defect', ['wrong_prompt','missing_correction','extra_correction','mixed_model','mixed_decoding'])
def test_incompatible_repairs_are_rejected(defect):
    old,new,prompts=fixture()
    if defect=='wrong_prompt': new=[old[1]]
    elif defect=='missing_correction': new=[]
    elif defect=='extra_correction': new.append(old[0])
    elif defect=='mixed_model': new[0]['model_revision']='different'
    elif defect=='mixed_decoding': new[0]['decoding']['max_new_tokens']=21
    with pytest.raises(ValueError): assemble_canonical(old,new,prompts,['b'])
