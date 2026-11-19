"""Download integrity checks without network or model weights."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location('alignlab_fetch_model', Path(__file__).parents[1] / 'scripts/fetch_model.py')
fetch = importlib.util.module_from_spec(spec)
with patch.dict('sys.modules', {'huggingface_hub': SimpleNamespace(HfApi=None, snapshot_download=None)}):
    spec.loader.exec_module(fetch)


def test_lfs_and_regular_files_are_verified(tmp_path):
    target = tmp_path / 'weights'
    target.write_bytes(b'abc')
    assert fetch.verify_file(target, 3, hashlib.sha256(b'abc').hexdigest())['bytes'] == 3
    assert fetch.verify_file(target, 3, git_blob=hashlib.sha1(b'blob 3\0abc').hexdigest())['bytes'] == 3


@pytest.mark.parametrize('size,lfs,blob', [(4, None, None), (3, '0' * 64, None), (3, None, '0' * 40)])
def test_corruption_or_truncation_rejected(tmp_path, size, lfs, blob):
    target = tmp_path / 'weights'
    target.write_bytes(b'abc')
    with pytest.raises(ValueError):
        fetch.verify_file(target, size, lfs, blob)
