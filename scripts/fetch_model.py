"""Explicit download of the pinned 7B model into the HF cache used by offline training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from huggingface_hub import HfApi, snapshot_download


def verify_file(path, expected_size, lfs_sha256=None, git_blob=None):
    path = Path(path)
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"Size mismatch: {path.name}")
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            sha.update(block)
            if not lfs_sha256:
                blob.update(block)
    if lfs_sha256 and sha.hexdigest() != lfs_sha256:
        raise ValueError(f"LFS checksum mismatch: {path.name}")
    if not lfs_sha256 and git_blob and blob.hexdigest() != git_blob:
        raise ValueError(f"Git blob mismatch: {path.name}")
    return {'bytes': size, 'sha256': sha.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-download', action='store_true')
    parser.add_argument('--max-workers', type=int, default=2)
    parser.add_argument('--manifest', default='runs/model_download/qwen7b.json')
    args = parser.parse_args()
    if not args.confirm_download:
        parser.error('7B weights require --confirm-download and sufficient disk space')
    if not 1 <= args.max_workers <= 8:
        parser.error('--max-workers must be between 1 and 8')
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / 'docs/sources.lock.json').read_text())['qwen7b']
    manifest = root / args.manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    record = {'status': 'running', 'model_id': source['id'], 'revision': source['revision'],
              'endpoint': os.environ.get('HF_ENDPOINT', 'https://huggingface.co'),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'files': {}}
    begin = time.monotonic()

    def save():
        temporary = manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        temporary.replace(manifest)

    save()
    try:
        api = HfApi()
        metadata = api.model_info(source['id'], revision=source['revision'], files_metadata=True)
        if metadata.sha != source['revision']:
            raise ValueError('Resolved model revision differs from the source lock')
        snapshot = Path(snapshot_download(source['id'], revision=source['revision'], max_workers=args.max_workers))
        for item in metadata.siblings:
            lfs = item.lfs
            expected = getattr(lfs, 'sha256', None) if lfs else None
            record['files'][item.rfilename] = verify_file(snapshot / item.rfilename, item.size, expected, item.blob_id)
            save()
        record.update(status='completed', snapshot_path=str(snapshot),
                      total_bytes=sum(f['bytes'] for f in record['files'].values()))
        print(snapshot, flush=True)
    except BaseException as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - begin
        record['ended_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
