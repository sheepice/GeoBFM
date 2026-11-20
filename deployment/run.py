#!/usr/bin/env python3
"""Export, verify and run simulation using the same camera and policy contract."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _bootstrap
from da2_vibe.cli import main

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[2] not in ("export", "verify", "sim2sim"):
        raise SystemExit("Usage: python deployment/run.py {stairs,stones} {export,verify,sim2sim} ...")
    raise SystemExit(main())
