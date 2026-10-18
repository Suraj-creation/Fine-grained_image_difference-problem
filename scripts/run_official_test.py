#!/usr/bin/env python
"""Run the UNMODIFIED official `test_ondata.py` from AndreaCodegoni/Tiny_model_4_CD.

The only thing added is a compatibility shim: the official `dataset/dataset.py` does
`from collections import Sized`, which was removed in Python 3.10 (the repo targets Python 3.8).
We restore the alias before the official code is imported. No official file is edited.

usage: python scripts/run_official_test.py --datapath data/levir \
           --modelpath third_party/Tiny_model_4_CD/pretrained_models/levir_best.pth
"""
import collections
import collections.abc
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "third_party" / "Tiny_model_4_CD"

collections.Sized = collections.abc.Sized  # py>=3.10 shim (see docstring)
sys.path.insert(0, str(REPO))
sys.argv = ["test_ondata.py"] + sys.argv[1:]
runpy.run_path(str(REPO / "test_ondata.py"), run_name="__main__")
