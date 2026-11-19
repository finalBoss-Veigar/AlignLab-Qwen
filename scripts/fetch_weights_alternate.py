"""Use ModelScope as transport, accepting only the exact pinned HF file hashes.

ModelScope master is NOT treated as experiment identity. Every file must match
the pinned HF metadata before it is exposed to training in the HF snapshot.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.parse import urlencode

from huggingface_hub import HfApi, snapshot_download
from fetch_model import verify_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-download', action='store_true')
    args = parser.parse_args()
    if not args.confirm_download:
        parser.error('--confirm-download is required')
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / 'docs/sources.lock.json').read_text())['qwen7b']
    run = root / 'runs/model_download' / datetime.now(timezone.utc).strftime('alternate-%Y%m%dT%H%M%SZ')
    run.mkdir(parents=True, exist_ok=False)
    transport = root / 'models/modelscope-transport' / source['revision']
    transport.mkdir(parents=True, exist_ok=True)
    record = {'status': 'running', 'model_id': source['id'], 'hf_revision': source['revision'],
              'transport': 'https://modelscope.cn', 'metadata_endpoint': os.environ.get('HF_ENDPOINT', 'https://huggingface.co'),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'files': {}}
    start = time.monotonic()

    def save():
        tmp = run / 'manifest.tmp'
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(run / 'manifest.json')

    save()
    try:
        info = HfApi().model_info(source['id'], revision=source['revision'], files_metadata=True)
        if info.sha != source['revision']:
            raise ValueError('Wrong HF revision')
        small = [f.rfilename for f in info.siblings if not f.rfilename.endswith('.safetensors')]
        snapshot = Path(snapshot_download(source['id'], revision=source['revision'], allow_patterns=small, max_workers=2))
        weights = [f for f in info.siblings if f.rfilename.endswith('.safetensors')]
        if len(weights) != 4:
            raise ValueError('Expected the four pinned Qwen7B weight shards')
        record['expected_weights'] = [{'path': f.rfilename, 'bytes': f.size, 'sha256': f.lfs.sha256} for f in weights]
        save()

        def fetch(item):
            target = snapshot / item.rfilename
            expected = item.lfs.sha256
            if target.exists():
                return item.rfilename, verify_file(target, item.size, expected)
            partial = transport / (item.rfilename + '.part')
            url = f"https://modelscope.cn/api/v1/models/{source['id']}/repo?" + urlencode({'Revision': 'master', 'FilePath': item.rfilename})
            subprocess.run(['curl', '-f', '-L', '--silent', '--show-error', '-C', '-', '--retry', '2',
                            '--connect-timeout', '15', '--max-time', '1500', '-o', str(partial), url], check=True)
            checked = verify_file(partial, item.size, expected)
            # Same filesystem hardlink keeps the transport copy without doubling disk use.
            os.link(partial, target)
            print(f'VERIFIED {item.rfilename} {expected}', flush=True)
            return item.rfilename, checked

        with ThreadPoolExecutor(max_workers=2) as pool:
            for name, checked in pool.map(fetch, weights):
                record['files'][name] = checked
                save()
        for item in info.siblings:
            if item.rfilename not in record['files']:
                record['files'][item.rfilename] = verify_file(snapshot / item.rfilename, item.size,
                    getattr(item.lfs, 'sha256', None) if item.lfs else None, item.blob_id)
        record.update(status='completed', snapshot_path=str(snapshot),
                      total_bytes=sum(x['bytes'] for x in record['files'].values()))
    except BaseException as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - start
        record['ended_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
