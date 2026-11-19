"""Copy immutable completed-stage archives over SSH and verify their SHA256 locally.

This helper never starts/stops training or modifies source data. Live snapshots are
explicitly snapshots, not completed experiment evidence. Credentials stay in SSH.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


REMOTE = r'''
import hashlib,json,os,tarfile,time
from pathlib import Path
root=Path(REMOTE_ROOT); os.chdir(root)
queue=root/'runs/queues'/QUEUE
state=json.loads((queue/'status.json').read_text())
backup=root/'runs/local_mirror_archives'; backup.mkdir(exist_ok=True)
entries=[]
def archive(name,files):
    dest=backup/(name+'.tar.gz')
    if not dest.exists():
        partial=dest.with_suffix('.partial')
        with tarfile.open(partial,'w:gz',compresslevel=1) as tar:
            for file in sorted(set(files)):
                if file.is_file() and not file.is_symlink():
                    tar.add(file,arcname=str(file.relative_to(root)),recursive=False)
        partial.replace(dest)
    sidecar=dest.with_suffix('.sha256')
    if sidecar.exists():
        sha=sidecar.read_text().strip()
    else:
        h=hashlib.sha256()
        with dest.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
        sha=h.hexdigest(); sidecar.write_text(sha)
    entries.append(dict(path=str(dest),name=dest.name,sha256=sha,bytes=dest.stat().st_size))
for name in ('base','sft','dpo','dpo_beta005'):
    score=root/'runs/formal_20261119'/(name+'_ifeval_score')
    if (score/'run.json').exists() and json.loads((score/'run.json').read_text())['status']=='completed':
        dirs=[p for p in (root/'runs/formal_20261119').iterdir() if p.is_dir() and
              p.name in (name+'_ifeval',name+'_ifeval_canonical',name+'_ifeval_repair',name+'_ifeval_score',name+'_preferences')]
        archive(name+'-evaluation', [f for d in dirs for f in d.rglob('*')])
for name in ('sft','dpo','dpo-beta005'):
    d=root/'runs/artifacts/gpu_20261118/formal'/name
    manifest=d/'alignlab_training_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()).get('status')=='completed':
        archive(name+'-training-checkpoints', list(d.rglob('*')))
# A unique live snapshot preserves logs/configs without copying changing weights.
files=[]
for d in ('runs/experiments','runs/queues','configs/gpu_20261118'):
    files += [p for p in (root/d).rglob('*') if p.suffix in ('.json','.jsonl','.yaml','.log','.txt')]
files += list((root/'runs').glob('recovery_*.json'))
for d in (root/'runs/artifacts/gpu_20261118/formal').glob('*'):
    if d.is_dir():files += [p for p in d.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl')]
archive('live-'+time.strftime('%Y%m%dT%H%M%S',time.gmtime()),files)
print(json.dumps(dict(queue=state,archives=entries,snapshot_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))))
'''


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--identity', required=True)
    parser.add_argument('--remote-root', default='/root/autodl-tmp/AlignLab')
    parser.add_argument('--queue', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--stop-at', required=True, help='ISO8601 timestamp including timezone')
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.stop_at.replace('Z', '+00:00'))
    if deadline.tzinfo is None or args.interval < 30:
        raise ValueError('Timezone and interval >=30 seconds required')
    dest = Path(args.destination); dest.mkdir(parents=True, exist_ok=True)
    record_path = dest / 'verified_backups.json'
    verified = json.loads(record_path.read_text()) if record_path.exists() else {}
    common = ['-i', args.identity, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
              '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3']
    remote_python = shlex.quote(args.remote_root + '/.venv-train/bin/python') + ' -'
    code = 'REMOTE_ROOT=' + repr(args.remote_root) + '\nQUEUE=' + repr(args.queue) + '\n' + REMOTE
    while datetime.now(timezone.utc) < deadline:
        try:
            response = subprocess.run(['ssh', *common, '-p', str(args.port), args.host, remote_python],
                                      input=code, text=True, capture_output=True, check=True, timeout=240)
            snapshot = json.loads(response.stdout)
            (dest / 'latest_snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
            print(json.dumps({'time': snapshot['snapshot_at'], 'queue_status': snapshot['queue']['status'],
                              'current': snapshot['queue']['current']}), flush=True)
            for item in snapshot['archives']:
                name = item['name']
                if Path(name).name != name or not name.endswith('.tar.gz'):
                    raise ValueError('Unsafe archive name')
                final = dest / name
                if name in verified and verified[name]['sha256'] == item['sha256'] and final.exists() and final.stat().st_size == item['bytes']:
                    continue
                if final.exists():
                    if checksum(final) != item['sha256']:
                        raise ValueError(f'Existing local archive differs; preserved: {final}')
                else:
                    partial = dest / (name + '.partial')
                    subprocess.run(['scp', *common, '-P', str(args.port),
                                    args.host + ':' + item['path'], str(partial)], check=True, timeout=1800)
                    if partial.stat().st_size != item['bytes'] or checksum(partial) != item['sha256']:
                        raise ValueError(f'Transfer hash mismatch: {partial}')
                    partial.replace(final)
                verified[name] = {**item, 'local_path': str(final.resolve()),
                                  'verified_at': datetime.now(timezone.utc).isoformat()}
                temporary = record_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(verified, indent=2) + '\n'); temporary.replace(record_path)
                print('VERIFIED ' + name, flush=True)
            if args.once or snapshot['queue']['status'] != 'running':
                return
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            print(json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'error': str(error)}), flush=True)
            if args.once:
                raise
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
