"""Exercise binary imports and a tiny Arrow/Datasets conversion; no model or GPU load."""
import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/training-imports.json')
    args = parser.parse_args()
    record = {'started_utc': datetime.now(timezone.utc).isoformat(), 'versions': {},
              'status': 'running', 'gpu_training_verified': False}
    try:
        modules = {}
        for name in ('numpy', 'pyarrow', 'torch', 'transformers', 'peft', 'trl',
                     'bitsandbytes', 'datasets', 'llamafactory'):
            module = importlib.import_module(name)
            modules[name] = module
            record['versions'][name] = getattr(module, '__version__', 'unknown')
        values = modules['numpy'].array([1, 2, 3], dtype='int64')
        table = modules['pyarrow'].table({'value': values})
        dataset = modules['datasets'].Dataset(table)
        if dataset['value'] != [1, 2, 3]:
            raise ValueError('Arrow/Datasets roundtrip mismatch')
        record.update(status='passed', arrow_dataset_roundtrip=True,
                      cuda_available=modules['torch'].cuda.is_available())
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record['ended_utc'] = datetime.now(timezone.utc).isoformat()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
