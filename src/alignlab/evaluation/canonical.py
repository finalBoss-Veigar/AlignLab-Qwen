"""Reuse only identical prompts; changed official prompts require new generations."""
from copy import deepcopy
from .generation import validate_generations
from .io import digest


def assemble_canonical(original, repairs, prompts, expected_corrections):
    validate_generations(original)
    validate_generations(repairs)
    old = {r['prompt_id']: r for r in original}
    new = {r['prompt_id']: r for r in repairs}
    wanted = {r['prompt_id']: r for r in prompts}
    if len(old) != len(original) or len(new) != len(repairs):
        raise ValueError('Assembly requires exactly one generation per prompt')
    if len(wanted) != len(prompts) or set(old) != set(wanted):
        raise ValueError('Canonical prompt coverage differs')
    changed = {key for key in wanted if old[key]['prompt'] != wanted[key]['prompt']}
    if changed != set(expected_corrections) or set(new) != changed:
        raise ValueError('Correction coverage must exactly match reviewed prompt changes')
    rows = []
    for prompt in prompts:
        key = prompt['prompt_id']
        source = new[key] if key in changed else old[key]
        context = prompt.get('messages') or [{'role': 'user', 'content': prompt['prompt']}]
        if source['prompt'] != prompt['prompt'] or source['context_sha256'] != digest(context):
            raise ValueError('A changed prompt cannot reuse the old response')
        row = deepcopy(source)
        row['assembly_origin'] = 'new_official_prompt_response' if key in changed else 'unchanged_prompt_response'
        row['source_generation_record_sha256'] = digest(source)
        row['original_input_source'] = deepcopy(source.get('source'))
        row['source'] = {name: prompt.get(name) for name in (
            'dataset_id', 'revision', 'source_row_id', 'prompt_hash', 'source_repository',
            'hf_source_revision', 'hf_source_row_id')}
        rows.append(row)
    validate_generations(rows)  # Reject mixed models, tokenizers or decoding controls.
    return rows
