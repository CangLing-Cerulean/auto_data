"""Preload supported libraries before placing generated code behind an audit hook."""
import argparse
import collections
import codecs
import io
import json
import math
from pathlib import Path
import statistics
import sys

import numpy
import torch

from .code_boundary import check_builder,install_boundary

def main(folder):
    # Exercise lazy ZIP/codec imports before restricting filesystem access.
    codecs.lookup('cp437')
    archive=io.BytesIO()
    numpy.savez_compressed(archive,preflight=numpy.array([1.0]))
    archive.seek(0)
    with numpy.load(archive,allow_pickle=False) as saved:
        saved['preflight']
    sample=numpy.array([1.0,2.0,2.0])
    numpy.unique(sample,return_counts=True)
    numpy.quantile(sample,[.1,.5,.9])
    numpy.percentile(sample,95)
    numpy.random.default_rng(0).permutation(3)
    numpy.fft.rfft(sample)
    numpy.linalg.svd(numpy.eye(2))
    folder=Path(folder).resolve()
    code=(folder/'code/agent.py').read_text(encoding='utf-8');check_builder(code)
    compiled=compile(code,str(folder/'code/agent.py'),'exec')
    inputs,outputs=folder/'inputs',folder/'outputs'
    sys.argv=[str(folder/'code/agent.py'),str(inputs),str(outputs)]
    violations=install_boundary(inputs,outputs)
    failed=False
    try:exec(compiled,{'__name__':'__main__','__file__':sys.argv[0]})
    except BaseException as error:
        # Formatting tracebacks with source lines would itself open forbidden files.
        failed=True
        cursor=error.__traceback__
        while cursor is not None:
            print(f'File "{cursor.tb_frame.f_code.co_filename}", line {cursor.tb_lineno}, in {cursor.tb_frame.f_code.co_name}',file=sys.stderr)
            cursor=cursor.tb_next
        print(type(error).__name__+': '+str(error),file=sys.stderr)
    finally:
        (outputs/'information_boundary.json').write_text(json.dumps({'violations':violations,
            'security_sandbox':False,'mechanism':'python_audit_and_observed_static_checks'}),encoding='utf-8')
    if failed:raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True)
    main(parser.parse_args().folder)
