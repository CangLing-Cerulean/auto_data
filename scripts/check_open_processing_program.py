"""Synthetic checks only; run with python scripts/check_open_processing_program.py."""

import unittest

import numpy as np

from open_processing_program import (
    compile_program, validate_measurement, validate_state, validate_target,
    validate_transform, validate_weights,
)


IDENTITY = '''
def fit(a):
    return {"center": np.mean(a, axis=0)}
def transform(x, state):
    return x, np.zeros(x.shape[1]), np.ones(x.shape[1])
def weights(x, state):
    return 1.0, 1.0
def target(x, state):
    return np.ones((8, x.shape[1]), dtype=np.bool_)
'''


class ProgramChecks(unittest.TestCase):
    def setUp(self):
        self.a = np.arange(60 * 4, dtype=float).reshape(60, 4) / 7
        self.x = np.arange(32 * 4, dtype=float).reshape(32, 4) / 11

    def compile_with(self, fit=None, transform=None):
        source = IDENTITY
        if fit:
            source = source.replace('return {"center": np.mean(a, axis=0)}', fit)
        if transform:
            source = source.replace('return x, np.zeros(x.shape[1]), np.ones(x.shape[1])', transform)
        return compile_program(source)

    def test_identity_and_a_only_state(self):
        program = self.compile_with()
        state = validate_state(program.call("fit", self.a))
        np.testing.assert_array_equal(state["center"], self.a.mean(axis=0))
        transformed, offset, scale = validate_transform(program.call("transform", self.x, state), channels=4)
        np.testing.assert_array_equal(transformed, self.x)
        np.testing.assert_array_equal(offset, np.zeros(4))
        np.testing.assert_array_equal(scale, np.ones(4))
        self.assertEqual(validate_weights(program.call("weights", self.x, state)), (1.0, 1.0))
        self.assertTrue(validate_target(program.call("target", self.x, state), channels=4).all())
        shifted = self.compile_with(transform='return x - state["center"], state["center"], np.ones(x.shape[1])')
        np.testing.assert_array_equal(shifted.call("transform", self.x, state)[0], self.x - self.a.mean(0))

    def test_linear_filter_center_scale_helpers_and_loops(self):
        source = '''
def smooth(x):
    y = x.copy()
    for channel in range(x.shape[1]):
        y[:, channel] = np.convolve(x[:, channel], np.ones(3) / 3, mode="same")
    return y
def fit(a):
    return {"mix": np.eye(a.shape[1]) + 0.1}
def transform(x, state):
    y = smooth(x @ state["mix"])
    center = y.mean(axis=0)
    scale = np.maximum(y.std(axis=0), 0.01)
    return (y - center) / scale, center, scale
def weights(x, state):
    total = 0.0
    i = 0
    while i < x.shape[1]:
        total += abs(x[0, i])
        i += 1
    return 1 + total, 2
def target(x, state):
    return np.array([[j % 2 == 0 for j in range(x.shape[1])] for i in range(8)])
'''
        program = compile_program(source)
        state = program.call("fit", self.a)
        result = validate_transform(program.call("transform", self.x, state), channels=4)
        mixed = self.x @ (np.eye(4) + 0.1)
        filtered = np.stack([np.convolve(mixed[:, i], np.ones(3) / 3, mode="same") for i in range(4)], axis=1)
        np.testing.assert_allclose(result[0], (filtered - filtered.mean(0)) / filtered.std(0))
        np.testing.assert_allclose(result[1], filtered.mean(0))
        np.testing.assert_allclose(result[2], filtered.std(0))
        self.assertEqual(validate_weights(program.call("weights", self.x, state)), (1 + abs(self.x[0]).sum(), 2))
        self.assertEqual(validate_target(program.call("target", self.x, state), channels=4).sum(), 16)

    def test_measurement_math_linalg_fft_and_dict(self):
        source = '''
def fit(a):
    return {"energy": np.linalg.norm(a), "n": a.shape[0]}
def measure(x, state):
    powers = np.abs(np.fft.rfft(x, axis=0))
    values = [v for k, v in state.items()]
    return math.sqrt(np.sum(powers ** 2)) / state.get("energy", 1)
'''
        program = compile_program(source, kind="measurement")
        state = program.call("fit", self.a)
        actual = validate_measurement(program.call("measure", self.x, state))
        expected = np.sqrt(np.sum(np.abs(np.fft.rfft(self.x, axis=0)) ** 2)) / np.linalg.norm(self.a)
        self.assertAlmostEqual(actual, expected)

    def test_loop_control_and_generator_comprehension(self):
        source = '''
def fit(a):
    return {}
def measure(x, state):
    total = 0
    for i in range(10):
        if i == 1:
            continue
        if i == 4:
            break
        total += i
    return total + sum(j * j for j in range(3))
'''
        self.assertEqual(compile_program(source, kind="measurement").call("measure", self.x, {}), 10)

    def test_independent_calls_and_copies(self):
        program = self.compile_with(transform='y = x.copy()\n    y[0] = 99\n    return y, state["center"].copy(), np.ones(x.shape[1])')
        state = program.call("fit", self.a)
        original = self.x.copy()
        first = program.call("transform", self.x, state)
        first[0][:] = -10
        second = program.call("transform", self.x, state)
        self.assertTrue((second[0][0] == 99).all())
        np.testing.assert_array_equal(self.x, original)
        self.assertTrue(second[0].flags.writeable)
        with self.assertRaises(ValueError):
            program.call("unknown", self.x)

    def test_readonly_arrays_and_nested_state(self):
        cases = [
            'x[0, 0] = 1\n    return x, np.zeros(4), np.ones(4)',
            'state["center"][0] = 1\n    return x, np.zeros(4), np.ones(4)',
            'state["new"] = 1\n    return x, np.zeros(4), np.ones(4)',
            'state["sequence"][0] = 1\n    return x, np.zeros(4), np.ones(4)',
        ]
        for body in cases:
            with self.subTest(body=body), self.assertRaises((TypeError, ValueError)):
                self.compile_with(transform=body).call("transform", self.x, {"center": np.zeros(4), "sequence": [0]})

    def test_compile_rejections(self):
        bodies = [
            'import os\n    return 0', 'return x.__class__', 'return np.load("x")',
            'return np.random.rand()', 'return open("x")', 'return eval("1")',
            'return getattr(x, "shape")', 'return type(x)', 'return (lambda: 1)()',
            'f = abs\n    return f(1)', 'return x.mean.__call__()',
            'return np.asarray(x).flags', 'np.pi = 1\n    return 0',
            'x.shape = (128,)\n    return 0', 'global value\n    return 0',
            'return transform(x, state)', 'return np.pad(x, 1, transform)',
            'np = x\n    return 0', 'abs = 1\n    return 0',
            'return _secret', 'return x.setflags(write=True)',
        ]
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.compile_with(transform=body)
        sources = [IDENTITY + '\nvalue = 3', IDENTITY.replace('def fit(a):', 'def fit(a=1):'),
                   IDENTITY.replace('def fit(a):', '@abs\ndef fit(a):'),
                   IDENTITY.replace('def fit(a):', 'def fit(a: int):'),
                   IDENTITY + '\ndef first(x):\n    return second(x)\ndef second(x):\n    return first(x)',
                   IDENTITY + '\ndef helper(np):\n    return 1',
                   IDENTITY + '\ndef helper(x):\n    def inner(y):\n        return y\n    return x']
        for source in sources:
            with self.subTest(source=source), self.assertRaises(ValueError):
                compile_program(source)
        with self.assertRaises(ValueError):
            compile_program(IDENTITY + '#' * 32768)
        with self.assertRaises(ValueError):
            compile_program(IDENTITY + '\ndef many(x):\n    return [' + ','.join('0' for _ in range(5100)) + ']')

    def test_object_dtype_and_namespace_mutability(self):
        for body in ['return np.array([1], dtype="object")', 'return x.astype("object")']:
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.compile_with(transform=body).call("transform", self.x, {})
        from open_processing_program import _namespace
        for proxy in (_namespace()["np"], _namespace()["np"].linalg, _namespace()["math"]):
            with self.assertRaises(TypeError):
                proxy.extra = 1

    def test_uninitialized_allocations_rejected(self):
        with self.assertRaises(ValueError):
            self.compile_with(transform='return np.empty_like(x)')
        for body in ['return np.pad(x, 1, mode="empty")', 'return np.pad(x, 1, "empty")',
                     'return np.pad(x, 0, "empty")']:
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, "uninitialized padding"):
                self.compile_with(transform=body).call("transform", self.x, {})
        for body in ['return np.pad(x, 1)', 'return np.pad(x, 1, "constant")',
                     'return np.pad(x, 1, mode="edge")']:
            with self.subTest(body=body):
                result = self.compile_with(transform=body).call("transform", self.x, {})
                self.assertEqual(result.shape, (34, 6))
                self.assertTrue(np.isfinite(result).all())

    def test_where_keyword_rejected_on_function_and_method_paths(self):
        for body in ['return np.sqrt(x, where=False)',
                     'return np.clip(x, 0, 1, where=False)',
                     'return x.clip(0, 1, where=False)',
                     'return np.sqrt(x, out=np.zeros_like(x), where=True)',
                     'return np.pad(x, 1, where=False)']:
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, "where keyword"):
                self.compile_with(transform=body).call("transform", self.x, {})
        result = self.compile_with(transform='return np.where(x > 1, x, -x)').call("transform", self.x, {})
        np.testing.assert_array_equal(result, np.where(self.x > 1, self.x, -self.x))

    def test_state_validation(self):
        state = {"a": self.a, "b": [1, (2.0, np.float32(3))]}
        copied = validate_state(state)
        copied["a"][0] = -1
        self.assertNotEqual(self.a[0, 0], -1)
        cycle = []
        cycle.append(cycle)
        deep = 0
        for _ in range(18):
            deep = [deep]
        for bad in [np.array([object()]), {1: 2}, float("nan"), np.array([np.inf]), np.timedelta64(1, "D"), "text", cycle, deep]:
            with self.subTest(bad_type=type(bad)), self.assertRaises(ValueError):
                validate_state(bad)
        with self.assertRaises(ValueError):
            validate_state(np.zeros(4 * 1024 * 1024 // 8 + 1))
        with self.assertRaises(ValueError):
            validate_state({"long-key": 1}, max_bytes=8)

    def test_state_structure_budget_boundaries(self):
        # Budgets cover empty nodes and sequence slots even with zero payload.
        for state, budget in [([], 64), ({}, 64), ((), 64), (np.zeros(0), 144),
                              ([[], {}], 208), ([np.zeros(0)], 216), ({"": []}, 208)]:
            with self.subTest(state_type=type(state), budget=budget):
                validate_state(state, max_bytes=budget)
                with self.assertRaisesRegex(ValueError, "byte budget"):
                    validate_state(state, max_bytes=budget - 1)
        for state in [[[] for _ in range(5)], [{} for _ in range(5)],
                      [np.zeros(0) for _ in range(3)]]:
            with self.subTest(), self.assertRaisesRegex(ValueError, "byte budget"):
                validate_state(state, max_bytes=256)

    def test_scalar_integer_and_dtype_width_boundaries(self):
        for value in [-(1 << 63), (1 << 63) - 1, np.int64(-(1 << 63)), np.uint64((1 << 63) - 1)]:
            with self.subTest(value=value):
                self.assertEqual(validate_state(value), int(value))
        for value in [-(1 << 63) - 1, 1 << 63, 1 << 1000, np.uint64(1 << 63)]:
            with self.subTest(value_type=type(value)), self.assertRaisesRegex(ValueError, "signed 64 bits"):
                validate_state(value)
        self.assertIs(type(validate_state(np.float64(1))), float)
        self.assertIs(type(validate_state(np.complex128(1 + 2j))), complex)
        for dtype, max_bytes in [(np.longdouble, 8), (np.clongdouble, 16)]:
            scalar = dtype(1)
            array = np.array([1], dtype=dtype)
            with self.subTest(dtype=dtype):
                if scalar.dtype.itemsize > max_bytes:
                    for value in [scalar, array]:
                        with self.assertRaisesRegex(ValueError, "dtype width"):
                            validate_state(value)
                else:
                    validate_state(scalar)
                    validate_state(array)
        with self.assertRaisesRegex(ValueError, "signed 64 bits"):
            self.compile_with(fit='return 1 << 1000').call("fit", self.a)

    def test_longdouble_scalar_state_ipc_roundtrip(self):
        from open_processing_sandbox import encode_state, pack, unpack, decode_state

        for dtype, number, max_bytes, python_type in [
                ("longdouble", 1.0, 8, float),
                ("clongdouble", 1.0 + 2.0j, 16, complex)]:
            scalar = np.array([number], dtype=dtype)[0]
            with self.subTest(dtype=dtype):
                if scalar.dtype.itemsize > max_bytes:
                    with self.assertRaisesRegex(ValueError, "dtype width"):
                        validate_state({"value": scalar})
                    continue
                state = validate_state({"value": scalar})
                self.assertIs(type(state["value"]), python_type)
                tree, arrays = encode_state(state)
                restored_tree, restored_arrays = unpack(pack(tree, arrays))
                restored = decode_state(restored_tree, restored_arrays)
                self.assertIs(type(restored["value"]), python_type)
                self.assertEqual(restored, {"value": number})

    def test_result_validation(self):
        valid = (self.x, np.zeros(4), np.ones(4))
        for bad in [list(valid), (self.x[:-1], valid[1], valid[2]), (self.x, np.zeros(3), valid[2]),
                    (self.x, valid[1], np.zeros(4)), (self.x, valid[1], -np.ones(4)),
                    (self.x * np.nan, valid[1], valid[2]), (self.x.astype(complex), valid[1], valid[2])]:
            with self.subTest(), self.assertRaises(ValueError):
                validate_transform(bad, channels=4)
        for bad in [(1, -1), (np.inf, 1), (np.array([1]), 1), (True, 1)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_weights(bad)
        for bad in [np.full((8, 4), 0.5), np.ones((7, 4)), np.full((8, 4), np.nan)]:
            with self.subTest(), self.assertRaises(ValueError):
                validate_target(bad, channels=4)
        for bad in [True, np.array(1.0), float("inf"), 1j, [1]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_measurement(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
