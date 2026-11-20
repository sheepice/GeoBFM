"""Prefer this checkout's code without installing into another experiment's environment."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DA2_VIBE_ROOT", str(ROOT))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
sys.dont_write_bytecode = True
paths = [ROOT / "src", ROOT / "training/src"]
paths += [ROOT / "third_party" / name / "src" for name in ("vibe", "orcs", "mocke")]
paths += [ROOT / "third_party/rsl_rl"]
sys.path[:0] = [str(p) for p in paths]
