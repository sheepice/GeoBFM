"""DA2 stairs with the existing vr-vibe three-point task and full motion mixture."""

import copy

from vibe_grail_joint.da2_flat_repair.task import (
    REMOVED_REWARDS,
    QualityStairsEnv,
    original_mixture,
    task_fractions,
)
from vibe_grail_joint.da2_flat_repair.task import (
    agent_config as base_agent_config,
)
from vibe_grail_joint.da2_flat_repair.task import (
    env_config as base_env_config,
)
from vibe_grail_joint.da2_flat_repair.task import (
    options as base_options,
)

from .camera import resolve_camera
from .task import configure_teleop_task

__all__ = [
    "REMOVED_REWARDS",
    "QualityStairsEnv",
    "agent_config",
    "env_config",
    "options",
    "original_mixture",
    "task_fractions",
]


def options(settings):
    result = base_options(settings)
    vr = settings["vr_teleop"]
    if not 0 <= vr["upper_probability"] <= 1 or vr["blend_seconds"] <= 0:
        raise ValueError("Invalid upper composition settings")
    if settings["visual"]["num_heads"] != 8:
        raise ValueError("Keep the inherited eight-head DA2 architecture")
    result["camera"] = resolve_camera(settings["camera"])
    result["vr_teleop"] = copy.deepcopy(vr)
    return result


def env_config(bank_path, settings, num_envs):
    options(settings)
    return configure_teleop_task(base_env_config(bank_path, settings, num_envs), settings)


def agent_config(bank_path, settings, num_envs):
    cfg = base_agent_config(bank_path, settings, num_envs)
    cfg["experiment_name"] = "vr_vibe_da2_teleop"
    cfg["actor"].update(
        class_name="vr_vibe.da2_model.DA2TeleopActor",
        teleop_checkpoint=settings["vr_teleop"]["encoder_checkpoint"],
        base_checkpoint=settings["dagger"]["teacher_checkpoint"],
    )
    cfg["algorithm"].update(
        class_name="vr_vibe.da2_algorithm.DA2TeleopPPO", vr_teleop=settings["vr_teleop"]
    )
    return cfg
