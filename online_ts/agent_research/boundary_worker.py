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
import traceback

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
    folder=Path(folder).resolve()
    code=(folder/'code/agent.py').read_text(encoding='utf-8');check_builder(code)
    compiled=compile(code,str(folder/'code/agent.py'),'exec')
    inputs,outputs=folder/'inputs',folder/'outputs'
    sys.argv=[str(folder/'code/agent.py'),str(inputs),str(outputs)]
    violations=install_boundary(inputs,outputs)
    failed=False
    try:exec(compiled,{'__name__':'__main__','__file__':sys.argv[0]})
    except BaseException:
        failed=True;traceback.print_exc()
    finally:
        (outputs/'information_boundary.json').write_text(json.dumps({'violations':violations,
            'security_sandbox':False,'mechanism':'python_audit_and_observed_static_checks'}),encoding='utf-8')
    if failed:raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True)
    main(parser.parse_args().folder)
