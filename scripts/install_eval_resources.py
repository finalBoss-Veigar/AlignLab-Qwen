"""Install the checksum-verified official NLTK resource without network access."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import zipfile

EXPECTED = 'e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    args = parser.parse_args()
    archive = Path(args.archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('NLTK archive checksum differs from the pinned official package')
    target = Path(__file__).resolve().parents[1] / 'models/nltk_data/tokenizers'
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            path = PurePosixPath(item.filename)
            if path.is_absolute() or '..' in path.parts or path.parts[0] != 'punkt_tab':
                raise ValueError('Unexpected archive path')
            if item.is_dir():
                continue
            content = zipped.read(item)
            destination = target.joinpath(*path.parts)
            if destination.exists():
                if destination.read_bytes() != content:
                    raise ValueError(f'Existing modified resource preserved: {destination}')
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    print(f'Verified NLTK punkt_tab installed under {target}')


if __name__ == '__main__':
    main()
