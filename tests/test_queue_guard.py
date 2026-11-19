from alignlab.training.queue_guard import matches_queue


def test_recognizes_legacy_queue_for_same_project(tmp_path):
    assert matches_queue(['python', 'scripts/run_full_experiments.py', '--execute'], tmp_path, tmp_path)


def test_other_project_does_not_block_this_project(tmp_path):
    assert not matches_queue(['python', 'scripts/run_full_experiments.py'], tmp_path/'other', tmp_path)


def test_shell_or_training_worker_not_mistaken_for_queue(tmp_path):
    assert not matches_queue(['bash','-c','python scripts/run_full_experiments.py'],tmp_path,tmp_path)
    assert not matches_queue(['python','-m','alignlab','train'],tmp_path,tmp_path)
