"""Validate and load agent-written, history-only, parameter-free input programs."""
import ast
import importlib.util
import json
import math
from pathlib import Path

from ..config import GateError


def processor_metadata(code_path, state_path):
    code_path,state_path=Path(code_path),Path(state_path)
    if any(p.is_symlink() or not p.is_file() for p in (code_path,state_path)):
        raise GateError('INPUT_PROCESSOR_FILES')
    if code_path.stat().st_size>16000 or state_path.stat().st_size>24000:
        raise GateError('INPUT_PROCESSOR_SIZE')
    source=code_path.read_text(encoding='utf-8')
    try:tree=ast.parse(source)
    except SyntaxError as error:raise GateError('INPUT_PROCESSOR_SYNTAX') from error
    if not any(isinstance(node,ast.FunctionDef) and node.name=='transform' for node in tree.body):
        raise GateError('INPUT_PROCESSOR_FUNCTION')
    forbidden={'open','eval','exec','compile','__import__','getattr','setattr','globals','locals',
               'load','save','load_state_dict','state_dict','nn','optim','hub','subprocess','socket','requests',
               'numpy','random','manual_seed','set_default_device','set_default_dtype','from_file'}
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
            if any(name not in ('torch','math') for name in names):raise GateError('INPUT_PROCESSOR_IMPORT')
        if isinstance(node,ast.Attribute) and (node.attr in forbidden or node.attr.startswith('_')):
            raise GateError('INPUT_PROCESSOR_EXTERNAL_EFFECT')
        if isinstance(node,ast.Name) and (node.id in forbidden or node.id.startswith('__')):
            raise GateError('INPUT_PROCESSOR_EXTERNAL_EFFECT')
        if isinstance(node,ast.keyword) and node.arg=='requires_grad':raise GateError('INPUT_PROCESSOR_TRAINABLE')
        if isinstance(node,(ast.Global,ast.Nonlocal)):raise GateError('INPUT_PROCESSOR_MUTABLE_GLOBAL')
    state=json.loads(state_path.read_text(encoding='utf-8'))
    def validate(value,depth=0):
        if depth>12:raise GateError('INPUT_PROCESSOR_STATE')
        if isinstance(value,dict):
            if any(not isinstance(k,str) for k in value):raise GateError('INPUT_PROCESSOR_STATE')
            for item in value.values():validate(item,depth+1)
        elif isinstance(value,list):
            for item in value:validate(item,depth+1)
        elif isinstance(value,float) and not math.isfinite(value):raise GateError('INPUT_PROCESSOR_STATE')
        elif value is not None and type(value) not in (int,float,str,bool):raise GateError('INPUT_PROCESSOR_STATE')
    if not isinstance(state,dict):raise GateError('INPUT_PROCESSOR_STATE')
    validate(state)
    return state


def load_processor(code_path,state_path):
    # Call only in credential-free validation/training subprocesses, never the controller.
    state=processor_metadata(code_path,state_path)
    spec=importlib.util.spec_from_file_location('agent_input_processor',code_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if not callable(module.transform):raise GateError('INPUT_PROCESSOR_FUNCTION')
    return module.transform,state


def apply_processor(x,program,*,identity_control=False):
    if program is None or identity_control:return x
    import torch
    function,state=program
    value=function(x,state)
    if not isinstance(value,torch.Tensor) or value.shape!=x.shape or value.device!=x.device or value.dtype!=x.dtype:
        raise GateError('INPUT_PROCESSOR_OUTPUT')
    if value.requires_grad or not torch.isfinite(value).all():raise GateError('INPUT_PROCESSOR_NONFINITE_OR_TRAINABLE')
    return value
