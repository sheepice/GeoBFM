#!/usr/bin/env python3
"""Build reference banks from explicit external motion and terrain resources."""
import _bootstrap
import argparse
import os
import runpy
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--paths")
args, rest = parser.parse_known_args()
if args.paths:
    os.environ["DA2_VIBE_PATHS"] = os.path.abspath(args.paths)
sys.argv = [sys.argv[0], *rest]
runpy.run_module("vibe_grail_joint.prepare", run_name="__main__")
