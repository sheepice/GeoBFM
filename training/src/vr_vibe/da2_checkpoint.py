"""Strict DA2-to-TELEOP actor transfer; the changed critic and losses start fresh."""
from da2_vibe.paths import load_checkpoint

import hashlib
import math
from pathlib import Path

import torch


def warm_start(actor, checkpoint, settings):
    path = Path(checkpoint)
    state = load_checkpoint(path, map_location="cpu", weights_only=False)
    metadata = state.get("infos", {}).get("da2_locomotion", {})
    if metadata.get("settings") != settings["visual"]:
        raise ValueError("Warm start requires identical DA2 weights, history and attention")
    camera = metadata.get("camera", {})
    for key in ("width", "height", "fovy", "fovx"):
        if key not in camera or not math.isclose(
            camera[key], settings["camera"][key], abs_tol=1e-8
        ):
            raise ValueError(f"DA2 camera mismatch: {key}")
    for key in ("parent_body", "pos", "quat"):
        if camera.get(key) != settings["camera"].get(key):
            raise ValueError(f"DA2 camera mount mismatch: {key}")
    missing, unexpected = actor.load_state_dict(state["actor_state_dict"], strict=False)
    expected = {k for k in actor.state_dict() if k.startswith("teleop_encoder.")}
    if unexpected or set(missing) != expected:
        raise ValueError(f"Unexpected DA2 actor mismatch: {missing}, {unexpected}")
    for name, value in state["actor_state_dict"].items():
        if not torch.equal(actor.state_dict()[name].cpu(), value):
            raise RuntimeError(f"Non-exact actor transfer: {name}")
    return {
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_iteration": int(state["iter"]),
        "inherited_actor_keys": len(state["actor_state_dict"]),
        "exact_all_da2_actor_parameters": True,
        "new_parameters": sorted(expected),
        "teleop_sha256": actor.teleop_weight_sha256,
        "fresh_critic_optimizer": True,
        "fresh_replay_buffers": True,
    }
