"""CPU-only generated-program worker with a Linux syscall allowlist.

The trusted parent supplies one authorized partition and fitted A state in a
NumPy archive on stdin. Generated code has no filesystem/network/process API.
This module is independent of the already running r10 experiment.
"""
import os
for _variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_variable]='1'

import argparse
import ctypes
import ctypes.util
import hashlib
import io
import json
import math
import platform
import signal
import socket
import sys
import time

import numpy as np


MAX_STATE_BYTES=4*1024*1024
ADDRESS_SPACE_BYTES=8*1024**3
CPU_SECONDS=120
WINDOW_SECONDS=1.
LENGTH,HORIZON,CHANNELS=32,8,370


def pack(metadata, arrays):
    stream=io.BytesIO()
    np.savez(stream,metadata=np.asarray(json.dumps(metadata,allow_nan=False)),**arrays)
    return stream.getvalue()


def unpack(payload):
    with np.load(io.BytesIO(payload),allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']))
        arrays={k:archive[k].copy() for k in archive.files if k!='metadata'}
    return metadata,arrays


def encode_state(state):
    arrays={}
    def encode(value):
        if isinstance(value,np.ndarray):
            key=f'state_{len(arrays)}';arrays[key]=np.array(value,copy=True)
            return dict(type='array',key=key)
        if isinstance(value,dict):return dict(type='dict',items={k:encode(v) for k,v in value.items()})
        if isinstance(value,(tuple,list)):return dict(type='sequence',items=[encode(v) for v in value])
        if isinstance(value,np.generic):value=value.item()
        if isinstance(value,complex):return dict(type='complex',real=value.real,imag=value.imag)
        return dict(type='scalar',value=value)
    return encode(state),arrays


def decode_state(tree, arrays):
    kind=tree['type']
    if kind=='array':return arrays[tree['key']].copy()
    if kind=='dict':return {k:decode_state(v,arrays) for k,v in tree['items'].items()}
    if kind=='sequence':return tuple(decode_state(v,arrays) for v in tree['items'])
    if kind=='scalar':return tree['value']
    if kind=='complex':return complex(tree['real'],tree['imag'])
    raise ValueError('Unknown state node')


def warm_libraries():
    # Resolve NumPy's lazy helpers before filesystem access is removed.
    unpack(pack(dict(warm=True),dict(x=np.arange(3.))))
    np.linalg.svd(np.eye(2));np.linalg.lstsq(np.eye(2),np.ones(2),rcond=None)
    np.linalg.eigh(np.eye(2));np.linalg.pinv(np.eye(2));np.fft.irfft(np.fft.rfft(np.arange(4.)),n=4)
    np.quantile(np.arange(4.),[.25,.75]);np.einsum('ij,jk->ik',np.eye(2),np.eye(2))


def install_limits():
    import resource
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
    resource.setrlimit(resource.RLIMIT_AS,(ADDRESS_SPACE_BYTES,ADDRESS_SPACE_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU,(CPU_SECONDS,CPU_SECONDS+1))


def install_filter():
    if platform.system()!='Linux' or platform.machine()!='x86_64':
        raise RuntimeError('The registered sandbox requires Linux x86_64')
    library=ctypes.util.find_library('seccomp')
    if not library:raise RuntimeError('libseccomp is required; no unsandboxed fallback')
    lib=ctypes.CDLL(library,use_errno=True)
    lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
    lib.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint];lib.seccomp_rule_add.restype=ctypes.c_int
    lib.seccomp_load.argtypes=[ctypes.c_void_p];lib.seccomp_load.restype=ctypes.c_int
    lib.seccomp_release.argtypes=[ctypes.c_void_p]
    libc=ctypes.CDLL(None,use_errno=True)
    libc.prctl.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong];libc.prctl.restype=ctypes.c_int
    if libc.prctl(38,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_NO_NEW_PRIVS failed')
    context=lib.seccomp_init(0x00050000|1)  # Default EPERM; allow only listed syscalls.
    if not context:raise MemoryError('seccomp_init failed')
    names=('read','write','close','fstat','lseek','mmap','mprotect','munmap','mremap','brk','madvise',
           'futex','rt_sigaction','rt_sigprocmask','rt_sigreturn','sigaltstack','setitimer',
           'clock_gettime','gettimeofday','time','getpid','gettid','sched_yield','sched_getaffinity',
           'getrandom','restart_syscall','getrusage','exit','exit_group')
    try:
        for name in names:
            number=lib.seccomp_syscall_resolve_name(name.encode())
            if number<0 or lib.seccomp_rule_add(context,0x7fff0000,number,0)!=0:
                raise RuntimeError(f'Cannot register syscall {name}')
        if lib.seccomp_load(context)!=0:raise RuntimeError('seccomp_load failed')
    finally:lib.seccomp_release(context)
    return names


def alarm_handler(signum,frame):
    raise TimeoutError('Generated function exceeded the per-call wall limit')


def timed_call(program,name,*arguments,seconds=WINDOW_SECONDS):
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:return program.call(name,*arguments)
    finally:signal.setitimer(signal.ITIMER_REAL,0.)


def checked_float32(value):
    with np.errstate(over='ignore',invalid='ignore'):
        result=np.asarray(value,dtype=np.float32)
    if not np.isfinite(result).all():raise ValueError('Result cannot be represented by finite float32 values')
    return result


def execute(metadata,arrays,engine):
    source=metadata['source'];kind=metadata['kind'];action=metadata['action']
    program=engine.compile_program(source,kind=kind)
    if action=='fit':
        rows=arrays['rows'];assert rows.ndim==2 and rows.shape[1]==CHANNELS and np.isfinite(rows).all()
        state=engine.validate_state(timed_call(program,'fit',rows.astype(np.float64),seconds=30.),max_bytes=MAX_STATE_BYTES)
        tree,values=encode_state(state)
        return dict(status='PASS',state=tree),values
    state=engine.validate_state(decode_state(metadata['state'],arrays),max_bytes=MAX_STATE_BYTES)
    rows=arrays['rows'];assert rows.ndim==2 and rows.shape[1]==CHANNELS and np.isfinite(rows).all()
    n=len(rows)-LENGTH-HORIZON+1;assert n>0
    if kind=='measurement':
        values=np.empty(n,dtype=np.float64)
        for i in range(n):
            values[i]=engine.validate_measurement(timed_call(program,'measure',rows[i:i+LENGTH].astype(np.float64),state))
        return dict(status='PASS',windows=n),dict(measurement=values)
    transformed=np.empty((n,LENGTH,CHANNELS),dtype=np.float32)
    offsets=np.empty((n,CHANNELS),dtype=np.float32);scales=np.empty_like(offsets)
    sampling=np.empty(n,dtype=np.float64);loss=np.empty(n,dtype=np.float64)
    masks=np.empty((n,HORIZON,CHANNELS),dtype=bool) if metadata['include_target'] else None
    for i in range(n):
        x=rows[i:i+LENGTH].astype(np.float64)
        encoded,offset,scale=engine.validate_transform(timed_call(program,'transform',x,state))
        transformed[i]=checked_float32(encoded);offsets[i]=checked_float32(offset);scales[i]=checked_float32(scale)
        if not (scales[i]>0).all():raise ValueError('Scale became zero in float32')
        if metadata['include_weights']:
            sampling[i],loss[i]=engine.validate_weights(timed_call(program,'weights',x,state))
        else:sampling[i]=loss[i]=1.
        if masks is not None:masks[i]=engine.validate_target(timed_call(program,'target',x,state))
    result=dict(model_input=transformed,offset=offsets,scale=scales,sampling=sampling,loss=loss)
    if masks is not None:result['target']=masks
    return dict(status='PASS',windows=n),result


def selfcheck():
    source_file=__file__
    warm_libraries();install_limits();names=install_filter()
    blocked={}
    for name,operation in (('file_open',lambda:open(source_file,'rb')),
                           ('network_socket',lambda:socket.socket()),
                           ('process_fork',lambda:os.fork())):
        try:
            value=operation()
            if name=='process_fork' and value==0:os._exit(99)
            if hasattr(value,'close'):value.close()
            blocked[name]=False
        except PermissionError:blocked[name]=True
    assert all(blocked.values()),blocked
    matrix=np.array([[2.,1.],[1.,2.]])
    singular=np.linalg.svd(matrix,compute_uv=False)
    assert np.allclose(singular,[3.,1.])
    assert np.allclose(np.fft.irfft(np.fft.rfft(np.arange(4.)),n=4),np.arange(4.))
    assert unpack(pack(dict(ok=True),dict(x=matrix)))[0]['ok']
    print(json.dumps(dict(status='PASS',blocked=blocked,numerical_operations=True,archive_io_in_memory=True,allowed_syscalls=list(names))))


def worker():
    import open_processing_program as engine
    metadata,arrays=unpack(sys.stdin.buffer.read())
    warm_libraries();signal.signal(signal.SIGALRM,alarm_handler)
    install_limits();install_filter();start=time.perf_counter()
    try:
        result,output=execute(metadata,arrays,engine)
        result.update(seconds=time.perf_counter()-start,source_sha256=hashlib.sha256(metadata['source'].encode()).hexdigest())
        payload=pack(result,output)
    except Exception as exc:
        result=dict(status='INVALID_PROGRAM',error=f'{type(exc).__name__}: {str(exc)[:4000]}',seconds=time.perf_counter()-start)
        payload=pack(result,{})
    sys.stdout.buffer.write(payload);sys.stdout.buffer.flush()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('selfcheck','worker'))
    globals()[parser.parse_args().action]()
