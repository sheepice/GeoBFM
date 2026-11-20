"""Resolve explicit resources and relocate metadata without changing source files."""

from functools import lru_cache
import json
import os
from pathlib import Path
import re


def repo_root():
    return Path(os.environ.get("DA2_VIBE_ROOT", Path(__file__).resolve().parents[2])).resolve()


def profile_path():
    return Path(os.environ.get("DA2_VIBE_PATHS", repo_root() / "configs/paths.local.json"))


@lru_cache(maxsize=8)
def _profile(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.is_file() else {}


def profile():
    return _profile(str(profile_path().resolve()))


def resolve(value):
    if isinstance(value, dict):
        return {key: resolve(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item) for item in value]
    if not isinstance(value, str):
        return value
    variables = dict(os.environ)
    variables.update(REPO=str(repo_root()),
                     ASSET_ROOT=os.environ.get("DA2_VIBE_ASSET_ROOT", str(repo_root() / "external")),
                     CACHE_ROOT=os.environ.get("DA2_VIBE_CACHE_ROOT", str(repo_root() / ".cache")))
    variables.update(profile().get("resources", {}))
    for _ in range(16):
        tokens = re.findall(r"\$\{([^}]+)\}", value)
        if not tokens:
            return os.path.expanduser(value)
        for token in tokens:
            if token not in variables:
                raise ValueError(f"Resource {token!r} is missing. Configure {profile_path()}")
            value = value.replace("${" + token + "}", str(variables[token]))
    raise ValueError("Resource definitions contain a cycle")


def remap_metadata(value):
    """Map path prefixes only; tensors, numerical settings and dict keys are unchanged."""
    if isinstance(value, dict):
        return {k: remap_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [remap_metadata(v) for v in value]
    if isinstance(value, tuple):
        return tuple(remap_metadata(v) for v in value)
    if isinstance(value, str):
        for old, new in sorted(profile().get("path_prefixes", {}).items(), key=lambda x: -len(x[0])):
            if value == old or value.startswith(old.rstrip("/") + "/"):
                return resolve(new).rstrip("/") + value[len(old.rstrip("/")):]
    return value


def load_settings(path):
    return remap_metadata(resolve(json.loads(Path(path).read_text())))


def load_checkpoint(path, *args, **kwargs):
    import torch
    return remap_metadata(torch.load(path, *args, **kwargs))
