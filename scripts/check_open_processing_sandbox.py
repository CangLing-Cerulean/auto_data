"""Synthetic Linux integration checks for the open-program worker."""
import json
import time
import numpy as np
import numpy.testing  # Load test helpers before the filesystem is sealed.
import open_processing_program as engine
import open_processing_sandbox as sandbox


PIPELINE='''
def fit(a):
    return {"mean": np.mean(a, axis=0), "spectrum": np.fft.rfft(a[:,0]), "scalar": 1j}
def transform(x, state):
    offset = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), 0.01)
    return (x-offset)/scale, offset, scale
def weights(x, state):
    return 1.0 + abs(np.mean(x)), 1.0
def target(x, state):
    return np.broadcast_to(np.arange(x.shape[1]) < 185, (8,x.shape[1]))
'''
MEASUREMENT='''
def fit(a):
    return {"mean": np.mean(a, axis=0)}
def measure(x, state):
    return np.mean((x-state["mean"])**2)
'''


def main():
    rng=np.random.default_rng(719)
    rows=rng.normal(size=(96,370)).astype(np.float32)
    sandbox.warm_libraries()
    import signal
    signal.signal(signal.SIGALRM,sandbox.alarm_handler)
    sandbox.install_limits();sandbox.install_filter()
    checks=[];start=time.perf_counter()
    fitted,state_arrays=sandbox.execute(dict(source=PIPELINE,kind='pipeline',action='fit'),dict(rows=rows),engine)
    state=sandbox.decode_state(fitted['state'],state_arrays)
    np.testing.assert_array_equal(state['mean'],rows.astype(np.float64).mean(0))
    assert state['scalar']==1j and np.iscomplexobj(state['spectrum'])
    metadata,arrays=sandbox.unpack(sandbox.pack(fitted,state_arrays))
    assert sandbox.decode_state(metadata['state'],arrays)['scalar']==1j
    checks.append('A_fit_numeric_state_complex_archive_roundtrip')
    transformed,out=sandbox.execute(dict(source=PIPELINE,kind='pipeline',action='apply',state=fitted['state'],include_target=True,include_weights=True),dict(rows=rows,**state_arrays),engine)
    assert transformed['windows']==57 and out['model_input'].shape==(57,32,370)
    for i in (0,28,56):
        original=rows[i:i+32].astype(np.float64)
        np.testing.assert_allclose(out['model_input'][i]*out['scale'][i]+out['offset'][i],original,rtol=2e-6,atol=2e-6)
        np.testing.assert_array_equal(out['target'][i,:,185:],False)
        np.testing.assert_array_equal(out['target'][i,:,:185],True)
    assert np.all(out['sampling']>=1.) and np.all(out['loss']==1.)
    checks.append('full_window_transform_inverse_weights_target')
    # A fitted state is passed to a different input partition without refitting.
    shifted=rows+100.
    _,again=sandbox.execute(dict(source=PIPELINE,kind='pipeline',action='apply',state=fitted['state'],include_target=False,include_weights=False),dict(rows=shifted,**state_arrays),engine)
    np.testing.assert_allclose(again['offset']-out['offset'],100.,atol=2e-5,rtol=0.)
    np.testing.assert_array_equal(sandbox.decode_state(fitted['state'],state_arrays)['mean'],state['mean'])
    checks.append('fitted_A_state_unchanged_on_other_partition')
    fitted_m,ma=sandbox.execute(dict(source=MEASUREMENT,kind='measurement',action='fit'),dict(rows=rows),engine)
    _,measured=sandbox.execute(dict(source=MEASUREMENT,kind='measurement',action='apply',state=fitted_m['state']),dict(rows=rows,**ma),engine)
    expected=np.array([np.mean((rows[i:i+32].astype(np.float64)-state['mean'])**2) for i in range(57)])
    np.testing.assert_array_equal(measured['measurement'],expected)
    checks.append('measurement_program_all_windows')
    endless=engine.compile_program('def fit(a):\n    while True:\n        pass\ndef measure(x,state):\n    return 0.0',kind='measurement')
    try:sandbox.timed_call(endless,'fit',rows,seconds=.1)
    except TimeoutError:checks.append('infinite_loop_timeout')
    else:raise AssertionError('Infinite loop was not stopped')
    oversized=engine.compile_program('def fit(a):\n    return np.ones(17179869184)\ndef measure(x,state):\n    return 0.0',kind='measurement')
    try:sandbox.timed_call(oversized,'fit',rows,seconds=1.)
    except MemoryError:checks.append('oversized_allocation_memory_limit')
    else:raise AssertionError('Oversized allocation was not stopped')
    print(json.dumps(dict(status='PASS',checks=checks,seconds=time.perf_counter()-start,real_data_access=False,full_trainings=0)))


if __name__=='__main__':main()
