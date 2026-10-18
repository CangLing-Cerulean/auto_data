"""Reviewed numeric Python subset; resource isolation belongs to the caller's OS sandbox."""

import ast
import builtins
import math
from types import MappingProxyType

import numpy as np


MAX_SOURCE_BYTES = 32768
MAX_AST_NODES = 5000
# Deterministic accounting includes tree structure, rather than only payloads.
STATE_CONTAINER_BYTES = 64
STATE_ARRAY_BYTES = 128
STATE_ARRAY_DIMENSION_BYTES = 16
STATE_SEQUENCE_SLOT_BYTES = 8
STATE_DICT_ENTRY_BYTES = 32
STATE_KEY_BYTES = 48
MIN_SCALAR_INTEGER = -(1 << 63)
MAX_SCALAR_INTEGER = (1 << 63) - 1
NP_FUNCTIONS = frozenset("array asarray copy zeros ones full zeros_like ones_like full_like arange linspace reshape transpose swapaxes moveaxis squeeze expand_dims stack concatenate broadcast_to tile repeat pad flip roll take where clip minimum maximum abs sign sqrt square exp log log1p expm1 tanh sin cos sum mean std var min max median quantile percentile diff cumsum cumprod sort argsort argmin argmax searchsorted unique count_nonzero all any isfinite isnan isinf dot matmul einsum convolve eye diag trace outer inner".split())
LINALG_FUNCTIONS = frozenset("norm svd eigh eigvalsh solve lstsq pinv".split())
FFT_FUNCTIONS = frozenset("rfft irfft rfftfreq".split())
MATH_FUNCTIONS = frozenset("sqrt log log1p exp sin cos tanh floor ceil".split())
DTYPES = frozenset("float64 float32 int64 int32 bool_".split())
BUILTINS = frozenset("abs min max sum len range enumerate zip float int bool round list tuple dict ValueError".split())
METHODS = frozenset("reshape transpose mean std var sum min max astype copy clip ravel flatten squeeze diagonal dot get keys values items".split())
READ_ATTRIBUTES = frozenset("shape ndim size T".split())


def _numeric_array(value):
    if value.dtype.kind not in "biufc":
        raise ValueError("only numeric, non-object arrays are allowed")
    if ((value.dtype.kind == "f" and value.dtype.itemsize > 8)
            or (value.dtype.kind == "c" and value.dtype.itemsize > 16)
            or (value.dtype.kind in "iu" and value.dtype.itemsize > 8)):
        raise ValueError("numeric arrays exceed supported dtype width")
    return value


def _numeric_scalar(value):
    if isinstance(value, np.generic):
        if value.dtype.kind not in "biufc":
            raise ValueError("only numeric scalars are allowed")
        if ((value.dtype.kind == "f" and value.dtype.itemsize > 8)
                or (value.dtype.kind == "c" and value.dtype.itemsize > 16)):
            raise ValueError("numeric scalars exceed supported dtype width")
        converters = {"b": bool, "i": int, "u": int, "f": float, "c": complex}
        value = converters[value.dtype.kind](value)
    if isinstance(value, int) and not MIN_SCALAR_INTEGER <= value <= MAX_SCALAR_INTEGER:
        raise ValueError("integer scalars must fit signed 64 bits")
    return value


def _copy_tree(value, readonly=False, finite=False, max_bytes=None):
    used = 0
    active = set()

    def charge(size):
        nonlocal used
        used += size
        if max_bytes is not None and used > max_bytes:
            raise ValueError("state exceeds byte budget")

    def visit(item, depth):
        if depth > 16:
            raise ValueError("tree depth exceeds 16")
        if isinstance(item, np.ndarray):
            _numeric_array(item)
            charge(STATE_ARRAY_BYTES + STATE_ARRAY_DIMENSION_BYTES * item.ndim + item.nbytes)
            if finite and not np.isfinite(item).all():
                raise ValueError("state contains non-finite array")
            result = item.copy()
            if readonly:
                result.flags.writeable = False
        elif isinstance(item, (bool, int, float, complex, np.number, np.bool_)):
            item = _numeric_scalar(item)
            if finite and not isinstance(item, int) and not np.isfinite(item):
                raise ValueError("state contains non-finite scalar")
            charge(max(8, (item.bit_length() + 7) // 8) if isinstance(item, int) else 16)
            result = item
        elif isinstance(item, (dict, MappingProxyType, tuple, list)):
            if id(item) in active:
                raise ValueError("cyclic trees are forbidden")
            entry_bytes = STATE_DICT_ENTRY_BYTES if isinstance(item, (dict, MappingProxyType)) else STATE_SEQUENCE_SLOT_BYTES
            charge(STATE_CONTAINER_BYTES + entry_bytes * len(item))
            active.add(id(item))
            try:
                if isinstance(item, (dict, MappingProxyType)):
                    result = {}
                    for key, child in item.items():
                        if not isinstance(key, str):
                            raise ValueError("dictionary keys must be strings")
                        charge(STATE_KEY_BYTES + len(key.encode("utf-8")))
                        result[key] = visit(child, depth + 1)
                    if readonly:
                        result = MappingProxyType(result)
                else:
                    result = [visit(child, depth + 1) for child in item]
                    if readonly or isinstance(item, tuple):
                        result = tuple(result)
            finally:
                active.remove(id(item))
        else:
            raise ValueError("only numeric scalars, arrays and dictionary/sequence trees are allowed")
        return result

    return visit(value, 0)


def validate_state(state, max_bytes=4 * 1024 * 1024):
    """Copy a numeric tree within its payload and deterministic structure budget."""
    if not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("max_bytes must be a nonnegative integer")
    return _copy_tree(state, finite=True, max_bytes=max_bytes)


def _real_array(value, shape):
    array = np.asarray(value)
    _numeric_array(array)
    if array.dtype.kind not in "biuf" or array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"expected finite real array with shape {shape}")
    return array.copy()


def validate_transform(result, channels=370, length=32):
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError("transform must return a 3-tuple")
    model_input = _real_array(result[0], (length, channels))
    offset = _real_array(result[1], (channels,))
    scale = _real_array(result[2], (channels,))
    if not (scale > 0).all():
        raise ValueError("scale must be strictly positive")
    return model_input, offset, scale


def validate_measurement(result):
    if not isinstance(result, (int, float, np.integer, np.floating)) or isinstance(result, (bool, np.bool_)):
        raise ValueError("measurement must be one finite real scalar")
    result = _numeric_scalar(result)
    try:
        number = float(result)
    except (OverflowError, ValueError) as error:
        raise ValueError("measurement must be finite") from error
    if not math.isfinite(number):
        raise ValueError("measurement must be finite")
    return number


def validate_weights(result):
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError("weights must contain sampling and loss scalars")
    weights = tuple(validate_measurement(item) for item in result)
    if min(weights) < 0:
        raise ValueError("weights must be nonnegative")
    return weights


def validate_target(result, horizon=8, channels=370):
    array = _real_array(result, (horizon, channels))
    if not ((array == 0) | (array == 1)).all():
        raise ValueError("target must contain only boolean or strict 0/1 values")
    return array


class _Proxy:
    __slots__ = ("_values",)

    def __init__(self, values):
        object.__setattr__(self, "_values", MappingProxyType(values))

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        raise TypeError("numeric namespace is immutable")


def _check_numeric_result(value):
    if isinstance(value, np.ndarray):
        _numeric_array(value)
    elif isinstance(value, (np.number, np.bool_)):
        _numeric_scalar(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _check_numeric_result(item)
    return value


def _wrapped(function):
    def invoke(*args, **kwargs):
        _reject_where(kwargs)
        return _check_numeric_result(function(*args, **kwargs))
    return invoke


def _reject_where(kwargs):
    if "where" in kwargs:
        raise ValueError("where keyword is forbidden; use positional np.where instead")


def _pad(array, pad_width, mode="constant", **kwargs):
    if mode == "empty":
        raise ValueError("uninitialized padding is forbidden")
    return _check_numeric_result(np.pad(array, pad_width, mode=mode, **kwargs))


def _namespace():
    values = {name: _wrapped(getattr(np, name)) for name in NP_FUNCTIONS}
    values["pad"] = _wrapped(_pad)
    values.update({name: getattr(np, name) for name in DTYPES})
    values.update(pi=np.pi, e=np.e)
    values["linalg"] = _Proxy({name: _wrapped(getattr(np.linalg, name)) for name in LINALG_FUNCTIONS})
    values["fft"] = _Proxy({name: _wrapped(getattr(np.fft, name)) for name in FFT_FUNCTIONS})
    math_values = {name: getattr(math, name) for name in MATH_FUNCTIONS | {"pi", "e"}}
    return {"__builtins__": {name: getattr(builtins, name) for name in BUILTINS},
            "np": _Proxy(values), "math": _Proxy(math_values), "_call_method": _call_method}


def _call_method(receiver, name, *args, **kwargs):
    array_methods = METHODS - {"get", "keys", "values", "items"}
    dict_methods = {"get", "keys", "values", "items"}
    if isinstance(receiver, np.ndarray) and name in array_methods:
        _reject_where(kwargs)
        return _check_numeric_result(getattr(receiver, name)(*args, **kwargs))
    if isinstance(receiver, (dict, MappingProxyType)) and name in dict_methods:
        return getattr(receiver, name)(*args, **kwargs)
    raise ValueError("method receiver must be a numeric array or dictionary")


def _path(node):
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _path(node.value)
        return parent + (node.attr,) if parent else None
    return None


def _static_path_allowed(path, callable_only=False):
    if not path:
        return False
    if len(path) == 2 and path[0] == "np":
        return path[1] in (NP_FUNCTIONS if callable_only else NP_FUNCTIONS | DTYPES | {"pi", "e", "linalg", "fft"})
    if len(path) == 3 and path[:2] == ("np", "linalg"):
        return path[2] in LINALG_FUNCTIONS
    if len(path) == 3 and path[:2] == ("np", "fft"):
        return path[2] in FFT_FUNCTIONS
    if len(path) == 2 and path[0] == "math":
        return path[1] in (MATH_FUNCTIONS if callable_only else MATH_FUNCTIONS | {"pi", "e"})
    return False


class _Review(ast.NodeVisitor):
    FORBIDDEN = frozenset("Import ImportFrom ClassDef Global Nonlocal With AsyncWith Try TryStar Yield YieldFrom AsyncFunctionDef Await Lambda Delete NamedExpr AnnAssign Match TypeAlias".split())

    def __init__(self, functions):
        self.functions = functions
        self.reserved = set(functions) | BUILTINS | {"np", "math"}
        self.current = None
        self.edges = {name: set() for name in functions}

    def generic_visit(self, node):
        if type(node).__name__ in self.FORBIDDEN:
            raise ValueError(f"forbidden syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node):
        if node.id.startswith("_"):
            raise ValueError("private names are forbidden")
        if isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in self.reserved:
            raise ValueError("cannot bind a reserved name")
        if isinstance(node.ctx, ast.Load) and node.id in self.functions:
            raise ValueError("helpers may only be used as direct call targets")

    def visit_FunctionDef(self, node):
        if self.current is not None:
            raise ValueError("nested functions are forbidden")
        args = node.args
        type_parameters = any(field == "type_params" and value for field, value in ast.iter_fields(node))
        if node.decorator_list or node.returns or type_parameters or args.defaults or args.kw_defaults or args.vararg or args.kwarg or args.kwonlyargs:
            raise ValueError("decorators, annotations, defaults and variable arguments are forbidden")
        for arg in args.posonlyargs + args.args:
            if arg.annotation or arg.arg.startswith("_") or arg.arg in self.reserved:
                raise ValueError("invalid function parameter")
        self.current = node.name
        for statement in node.body:
            self.visit(statement)
        self.current = None

    def visit_Attribute(self, node):
        if node.attr.startswith("_") or not isinstance(node.ctx, ast.Load):
            raise ValueError("private attributes and attribute writes are forbidden")
        path = _path(node)
        if path and path[0] in {"np", "math"}:
            if not _static_path_allowed(path):
                raise ValueError("numeric namespace attribute is not allowed")
        elif node.attr not in READ_ATTRIBUTES:
            raise ValueError("object attributes may only be read from the approved list")
        self.visit(node.value)

    def visit_Call(self, node):
        target = node.func
        if isinstance(target, ast.Name):
            if target.id not in BUILTINS and target.id not in self.functions:
                raise ValueError("dynamic calls are forbidden")
            if target.id in self.functions:
                self.edges[self.current].add(target.id)
        elif isinstance(target, ast.Attribute):
            path = _path(target)
            if path and path[0] in {"np", "math"}:
                if not _static_path_allowed(path, callable_only=True):
                    raise ValueError("numeric function is not allowed")
                self.visit(target)
            elif target.attr in METHODS:
                self.visit(target.value)
            else:
                raise ValueError("object method is not allowed")
        else:
            raise ValueError("calls on returned objects are forbidden")
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg.startswith("_"):
                raise ValueError("expanded/private keyword arguments are forbidden")
            self.visit(keyword.value)


class _Methods(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute):
            path = _path(node.func)
            if not (path and path[0] in {"np", "math"}):
                node.args = [node.func.value, ast.Constant(node.func.attr)] + node.args
                node.func = ast.Name(id="_call_method", ctx=ast.Load())
        return node


class CompiledProgram:
    def __init__(self, code, functions, kind):
        self.code = code
        self.functions = frozenset(functions)
        self.kind = kind

    def call(self, name, *args):
        if name not in self.functions:
            raise ValueError("call target is not a defined function")
        namespace = _namespace()
        exec(self.code, namespace)
        inputs = tuple(_copy_tree(argument, readonly=True) for argument in args)
        return _copy_tree(namespace[name](*inputs))


def compile_program(source, kind="pipeline"):
    if kind not in {"pipeline", "measurement"}:
        raise ValueError("unknown program kind")
    if not isinstance(source, str) or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("source must be UTF-8 text within 32768 bytes")
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, RecursionError) as error:
        raise ValueError("invalid program syntax") from error
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError("program exceeds 5000 AST nodes")
    functions = {}
    for index, statement in enumerate(tree.body):
        if index == 0 and isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            continue
        if not isinstance(statement, ast.FunctionDef):
            raise ValueError("top level permits only functions and a module docstring")
        if statement.name.startswith("_") or statement.name in functions or statement.name in BUILTINS | {"np", "math"}:
            raise ValueError("invalid or duplicate function name")
        functions[statement.name] = statement
    required = {"fit": 1, "measure": 2} if kind == "measurement" else {"fit": 1, "transform": 2, "weights": 2, "target": 2}
    for name, arity in required.items():
        if name not in functions or len(functions[name].args.posonlyargs + functions[name].args.args) != arity:
            raise ValueError(f"required function {name} must accept {arity} arguments")
    review = _Review(functions)
    review.visit(tree)
    visiting = set()
    visited = set()
    def walk(name):
        if name in visiting:
            raise ValueError("recursive helper calls are forbidden")
        if name in visited:
            return
        visiting.add(name)
        for child in review.edges[name]:
            walk(child)
        visiting.remove(name)
        visited.add(name)
    for name in functions:
        walk(name)
    tree = ast.fix_missing_locations(_Methods().visit(tree))
    return CompiledProgram(compile(tree, "<numeric-program>", "exec"), functions, kind)
