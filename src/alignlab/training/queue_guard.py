"""Single-host queue exclusion, including queues launched before lock support."""
import os
from pathlib import Path


def matches_queue(argv, cwd, root):
    return (len(argv) >= 2 and Path(argv[1]).name == 'run_full_experiments.py'
            and Path(cwd).resolve() == Path(root).resolve())


def acquire_queue_lock(root):
    import fcntl  # Linux executor only; not imported by offline Windows checks.
    root = Path(root).resolve()
    lock_path = root / 'runs/.full_experiment_queue.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another experiment queue holds this project lock') from None
        # Older running queues have no file lock; recognize their actual process.
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            try:
                argv = entry.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
                cwd = entry.joinpath('cwd').resolve(strict=True)
            except (OSError, UnicodeError):
                continue
            if matches_queue(argv, cwd, root):
                raise RuntimeError(f'Existing experiment queue PID {entry.name}; inspect it before restarting')
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid()) + '\n'); lock.flush()
        return lock  # Caller retains this open handle for the queue lifetime.
    except BaseException:
        lock.close()
        raise
