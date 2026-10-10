"""Bounded native Linux jobs; user-authorized direct execution, not a sandbox."""
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

from ..config import GateError, digest
from ..selection.common import file_hash, read_json, write_json


def inventory(folder, maximum_bytes, maximum_files):
    total, files = 0, []
    for path in Path(folder).rglob('*'):
        if path.is_symlink():
            raise GateError('NATIVE_OUTPUT_LINK')
        if path.is_dir():
            continue
        if not path.is_file():
            raise GateError('NATIVE_OUTPUT_TYPE')
        total += path.stat().st_size
        files.append(path)
        if total > maximum_bytes or len(files) > maximum_files:
            raise GateError('NATIVE_OUTPUT_LIMIT')
    return files, total


def run_process(command, *, cwd, outputs, seconds, memory_bytes=None,
                max_output_bytes=64*1024**2, max_files=32, log_bytes=65536, gpu=False):
    """Limits are process/monitor limits; total output checking is not a hard quota."""
    if sys.platform != 'linux':
        raise GateError('SERVER_LINUX_REQUIRED')
    import resource
    def limits():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32*1024**2, 32*1024**2))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        resource.setrlimit(resource.RLIMIT_CPU, (int(seconds)+1, int(seconds)+1))
        if memory_bytes is not None:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    env = {'PATH': '/opt/miniconda3/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8',
           'HOME': str(outputs), 'TMPDIR': str(outputs), 'OPENBLAS_NUM_THREADS': '1',
           'OMP_NUM_THREADS': '4' if gpu else '1', 'MKL_NUM_THREADS': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    if not gpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    started, captured, error = time.monotonic(), bytearray(), None
    process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=limits)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while True:
            for key, _ in selector.select(.05):
                block = os.read(key.fileobj.fileno(), 4096)
                if not block:
                    selector.unregister(key.fileobj)
                else:
                    captured.extend(block[:max(0, log_bytes-len(captured))])
                    if len(captured) >= log_bytes:
                        raise GateError('NATIVE_LOG_LIMIT')
            inventory(outputs, max_output_bytes, max_files)
            if time.monotonic()-started > seconds:
                raise GateError('NATIVE_TIMEOUT')
            if process.poll() is not None and not selector.get_map():
                break
    except Exception as failure:
        error = str(failure) if isinstance(failure, GateError) else type(failure).__name__
    finally:
        # Kill the whole ordinary process group, including children left by a
        # finished parent. This is not protection against adversarial setsid().
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        selector.close()
        process.stdout.close()
    try:
        files, size = inventory(outputs, max_output_bytes, max_files)
    except GateError as failure:
        files, size, error = [], None, str(failure)
    text = captured.decode('utf-8', errors='replace')
    secret = os.environ.get('DEEPSEEK_API_KEY')
    if secret:
        text = text.replace(secret, '[REDACTED]')
    return {'status': 'completed' if error is None and process.returncode == 0 else 'failed',
            'returncode': process.returncode, 'error': error, 'stdout': text,
            'elapsed_seconds': time.monotonic()-started, 'output_bytes': size,
            'files': [p.relative_to(outputs).as_posix() for p in files],
            'security_sandbox': False, 'total_output_limit': 'polled_and_checked_after_exit'}


def execute_job(folder, guards, *, seconds=30):
    folder = Path(folder)
    job = read_json(folder / 'job.json')
    if digest({k:v for k,v in job.items() if k!='sha256'}) != job['sha256']:
        raise GateError('NATIVE_JOB_IDENTITY')
    assets = {str(folder/'code/agent.py'): job['code_sha256']}
    assets.update({str(folder/'inputs'/a['name']): a['sha256'] for a in job['inputs']})
    expected = {**guards, **assets}
    for path, value in expected.items():
        if file_hash(path) != value:
            raise GateError('NATIVE_PROTECTED_ASSET_CHANGED')
    receipt_path = folder / 'execution.json'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt['job_sha256'] != job['sha256'] or receipt['status'] != 'completed':
            raise GateError('NATIVE_FAILED_JOB_NOT_REPLAYED')
        for name, value in receipt['output_sha256'].items():
            if file_hash(folder/'outputs'/name) != value:
                raise GateError('NATIVE_CACHED_OUTPUT_CHANGED')
        return receipt
    if (folder/'started.json').exists():
        raise GateError('NATIVE_INTERRUPTED_JOB_NOT_REPLAYED')
    write_json(folder/'started.json', {'job_sha256': job['sha256']})
    result = run_process([sys.executable, '-B', str(folder/'code/agent.py'),
                          str(folder/'inputs'), str(folder/'outputs')],
                         cwd=folder/'outputs', outputs=folder/'outputs', seconds=seconds,
                         memory_bytes=8*1024**3)
    changed = [path for path, value in expected.items() if not Path(path).is_file() or file_hash(path)!=value]
    result.update(job_sha256=job['sha256'], protected_assets_changed=changed)
    if changed:
        result.update(status='failed', error='NATIVE_PROTECTED_ASSET_CHANGED')
    result['output_sha256'] = {name:file_hash(folder/'outputs'/name) for name in result['files']}
    write_json(receipt_path, result)
    if changed:
        raise GateError('NATIVE_PROTECTED_ASSET_CHANGED')
    return result
