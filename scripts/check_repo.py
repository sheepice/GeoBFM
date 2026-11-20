#!/usr/bin/env python3
"""Audit tracked source for binary artifacts, machine paths and missing Python modules."""
import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {".pt", ".pth", ".ckpt", ".onnx", ".bin", ".npz", ".npy", ".pkl", ".safetensors", ".mp4", ".gif"}


def main():
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    paths = [ROOT / name for name in names if name]
    errors = []
    for path in paths:
        if path.suffix in FORBIDDEN or path.stat().st_size > 1_000_000:
            errors.append(f"Large/binary artifact: {path.relative_to(ROOT)}")
        if path.is_symlink():
            errors.append(f"Tracked symlink: {path.relative_to(ROOT)}")
        if path.name.endswith(".local.json"):
            errors.append(f"Local resource profile is tracked: {path.name}")
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
        if path.suffix in (".py", ".json", ".toml", ".sh") and "third_party" not in path.parts:
            text = path.read_text()
            if any(prefix in text for prefix in ("/".join(("", "mnt", "workspace", "")), "/".join(("", "home", "wjw", "")))):
                errors.append(f"Machine-specific source/config: {path.relative_to(ROOT)}")
    modules = {}
    for folder in (ROOT / "src", ROOT / "training/src"):
        for path in folder.rglob("*.py"):
            name = ".".join(path.relative_to(folder).with_suffix("").parts)
            modules[name.removesuffix(".__init__")] = path
    for name, path in modules.items():
        package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = node.module or ""
            if node.level:
                target = ".".join(package.split(".")[:len(package.split(".")) - node.level + 1]) + ("." + target if target else "")
            if target.startswith(("vibe_grail_joint", "vr_vibe", "da2_vibe")) and target not in modules:
                errors.append(f"Missing internal dependency: {name}:{node.lineno} -> {target}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"PASS: {len(paths)} tracked files, {sum(p.stat().st_size for p in paths) / 2**20:.2f} MiB; no weights/data/local profiles or missing internal imports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
