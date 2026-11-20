"""Causal three-point observations and rewards shared by DA2 TELEOP tasks."""
from dataclasses import fields
import torch
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from .reference import VRMotionCommandCfg
from .contracts import inv, mul, teleop_input, points_from_bodies

def teleop_observation(env):
    c = env.command_manager.get_term("motion")
    ids = c.future_frames(10, 5)
    lower_pos = c.motion.joint_pos[ids][..., :12]
    lower_vel = c.motion.joint_vel[ids][..., :12]
    relative = mul(inv(c.robot.data.root_link_quat_w), c.motion.body_quat_w[c.time_steps, 0])
    return teleop_input(lower_pos, lower_vel, c.vr_points, c.vr_orientations, relative)

def reference_mode(env):
    return env.command_manager.get_term("motion").online.float()[:, None]

def actual_points(c):
    return points_from_bodies(
        c.robot_body_pos_w,
        c.robot_body_quat_w,
        c.cfg.body_names,
        c.robot.data.root_link_pos_w,
        c.robot.data.root_link_quat_w,
    )

def point_tracking(env, kind, std):
    c = env.command_manager.get_term("motion")
    p, q = actual_points(c)
    if kind == "position":
        error = (p - c.vr_points).square().sum(-1)
    else:
        # Quaternion distance is sign-invariant; orientation convention matches release.
        angle = 2 * torch.acos((q * c.vr_orientations).sum(-1).abs().clamp(max=1.0))
        error = angle.square()
    return torch.exp(-error / std**2).mean(-1) * c.ppo_online

def configure_teleop_task(cfg, settings):
    """Add the same causal targets and rewards to either visual backbone."""
    old = cfg.commands["motion"]
    params = {
        f.name: getattr(old, f.name) for f in fields(VRMotionCommandCfg) if hasattr(old, f.name)
    }
    params.update(
        vr_upper_probability=settings["vr_teleop"]["upper_probability"],
        vr_upper_sources=tuple(settings["vr_teleop"]["upper_sources"]),
        vr_blend_seconds=settings["vr_teleop"]["blend_seconds"],
    )
    cfg.commands["motion"] = VRMotionCommandCfg(**params)
    for name, func in (
        ("teleop_reference", teleop_observation),
        ("reference_mode", reference_mode),
    ):
        cfg.observations[name] = ObservationGroupCfg(
            terms={"value": ObservationTermCfg(func=func)},
            concatenate_terms=True,
            enable_corruption=False,
        )
    # The value function must observe the task target whose tracking reward it predicts.
    cfg.observations["critic"].terms["teleop_reference"] = ObservationTermCfg(
        func=teleop_observation
    )
    cfg.observations["critic"].terms["reference_mode"] = ObservationTermCfg(func=reference_mode)
    # Three-point targets do not uniquely specify waist/arm joint angles. The input
    # reference is the primary target; original action prior remains a weak secondary loss.
    for name in (
        "online_waist_position",
        "online_arms_position",
        "online_waist_velocity",
        "online_arms_velocity",
        "online_torso_orientation",
    ):
        cfg.rewards.pop(name, None)
    vr = settings["vr_teleop"]
    cfg.rewards["vr_point_position"] = RewardTermCfg(
        func=point_tracking,
        weight=vr["position_weight"],
        params=dict(kind="position", std=vr["position_std_m"]),
    )
    cfg.rewards["vr_point_orientation"] = RewardTermCfg(
        func=point_tracking,
        weight=vr["orientation_weight"],
        params=dict(kind="orientation", std=vr["orientation_std_rad"]),
    )
    return cfg
