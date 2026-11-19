"""Fetch only the pinned official IFEval source; no model, dataset or scoring run."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    if sys.platform != 'linux':
        raise RuntimeError('Run on Linux: the upstream monorepo has Windows-incompatible paths')
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / 'docs/sources.lock.json').read_text(encoding='utf-8'))['ifeval_code']
    revision = source['revision']
    checkout = root / 'vendor/google-research'

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=checkout, text=True).strip()

    if checkout.exists():
        if not (checkout / '.git').is_dir():
            raise RuntimeError('Existing source directory preserved; expected a Git checkout')
    else:
        checkout.mkdir(parents=True)
        git('init')
        git('remote', 'add', 'origin', 'https://github.com/google-research/google-research.git')
        git('config', 'remote.origin.promisor', 'true')
        git('config', 'remote.origin.partialclonefilter', 'blob:none')
        git('sparse-checkout', 'init', '--cone')
        git('sparse-checkout', 'set', 'instruction_following_eval')
        git('fetch', '--depth', '1', '--filter=blob:none', 'origin', revision)
        git('checkout', '--detach', 'FETCH_HEAD')
    actual = git('rev-parse', 'HEAD')
    if actual != revision:
        raise RuntimeError('Existing source differs from pin; no reset performed')
    if git('status', '--porcelain', '--', 'instruction_following_eval'):
        raise RuntimeError('Official IFEval source has local changes; preserved')
    for relative in ('evaluation_main.py', 'data/input_data.jsonl', 'requirements.txt'):
        if not (checkout / 'instruction_following_eval' / relative).is_file():
            raise RuntimeError(f'Incomplete official checkout: {relative}')
    print(json.dumps({'status': 'source_verified', 'revision': actual,
                      'path': str(checkout), 'model_evaluation_completed': False}, indent=2))


if __name__ == '__main__':
    main()
