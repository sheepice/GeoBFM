from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _bootstrap
import mjlab  # register simulator dependencies before importing task configurations
