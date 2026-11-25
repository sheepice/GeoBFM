"""A single free-object carry with actor-visible reference and current object state."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_mul, quat_inv, matrix_from_quat
from mocke.mdp.commands import FutureMotionCommand, FutureMotionCommandCfg
from mocke.sonic.env_cfg import sonic_tracking_env_cfg
from orcs.core.obs import proprio_terms, robot_root_twist_cmd_terms
from vibe.core.sensors import head_cam_cfg
from vibe_grail_joint.da2_locomotion.task import configure_visual_env


def motion(env):
    return env.command_manager.get_term("motion")


class CarryCommand(FutureMotionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.box = env.scene["box"]
        with np.load(cfg.motion_file) as src:
            for k in ("box_pose", "box_lin_vel", "box_ang_vel", "palms", "grip", "vertices"):
                setattr(self, k, torch.as_tensor(src[k], device=self.device, dtype=torch.float32))
            forward = src["box_pose"][0, :2] - src["qpos"][0, :2]
            forward /= np.linalg.norm(forward)
            self.box_axes = torch.tensor(
                [forward.tolist(), [-forward[1], forward[0]]], device=self.device, dtype=torch.float32
            )
        self.first_grip = int((self.grip.max(dim=1).values > 0).nonzero()[0].item())
        self.fresh = torch.zeros_like(self.time_steps, dtype=torch.bool)
        self.full = self.fresh.clone()
        self.lift = self.fresh.clone()
        self.airborne = self.fresh.clone()
        self.carried = self.fresh.clone()
        self.success = self.fresh.clone()
        self.air_start = torch.zeros((self.num_envs, 2), device=self.device)
        self.stable = torch.zeros_like(self.time_steps)
        self.counts = torch.zeros(6 if cfg.smooth_grasp else 4, device=self.device, dtype=torch.long)
        if cfg.smooth_grasp:
            self.nonhand_contact = self.fresh.clone()
            self.nonhand_peak_n = torch.zeros(self.num_envs, device=self.device)
        if cfg.visual_pickup:
            self.calibration = torch.load(
                Path(cfg.motion_file).with_name("pickup_vision.pt"),
                map_location=self.device,
                weights_only=True,
            )
            self.pickup_offset = torch.zeros((self.num_envs, 3), device=self.device)
            self.last_pickup_tick = torch.full_like(self.time_steps, -1)
            self.far_start = self.fresh.clone()
            self.far_counts = torch.zeros(4, device=self.device, dtype=torch.long)
        self.sites = [self.robot.site_names.index(s + "_palm") for s in ("left", "right")]

    def _uniform_sampling(self, env_ids):
        draw = torch.rand(len(env_ids), device=self.device)
        # Full starts and pre-grasp states are the majority; later starts train carrying and release.
        late = torch.randint(0, self.motion.time_step_total - 34, (len(env_ids),), device=self.device)
        first_contact = int(self.first_grip)
        early = torch.randint(
            max(1, first_contact - 30), max(2, first_contact + 1), (len(env_ids),), device=self.device
        )
        if self.cfg.grasp_feedback:
            grip_phase = torch.randint(
                first_contact, self.motion.time_step_total - 35, (len(env_ids),), device=self.device
            )
            self.time_steps[env_ids] = torch.where(
                draw < (0.65 if self.cfg.visual_pickup else 0.30),
                0,
                torch.where(draw < (0.80 if self.cfg.visual_pickup else 0.65), early, grip_phase),
            )
        else:
            self.time_steps[env_ids] = torch.where(draw < 0.55, 0, torch.where(draw < 0.8, early, late))

    def _resample_command(self, env_ids):
        if self.cfg.visual_pickup:
            self.pickup_offset[env_ids] = 0
            self.last_pickup_tick[env_ids] = -1
            self.far_start[env_ids] = False
        super()._resample_command(env_ids)
        t = self.time_steps[env_ids]
        pose = self.box_pose[t].clone()
        pose[:, :3] += self._env.scene.env_origins[env_ids]
        if self.cfg.visual_pickup and self.cfg.sampling_mode == "uniform":
            # Oversample farther starts and retain original placements for comparison.
            uv = torch.rand(len(t), 2, device=self.device)
            group = torch.rand(len(t), device=self.device)
            forward = torch.where(group < 0.35, 0.10 + 0.20 * uv[:, 0], -0.12 + 0.42 * uv[:, 0])
            offset = torch.stack([forward, (2 * uv[:, 1] - 1) * 0.18], dim=-1)
            offset *= ((group < 0.80) & (t == 0))[:, None]
            pose[:, :2] += offset @ self.box_axes
            self.far_start[env_ids] = offset[:, 0] >= 0.10
        elif any(self.cfg.box_jitter_m):
            offset = (torch.rand(len(t), 2, device=self.device) * 2 - 1) * pose.new_tensor(
                self.cfg.box_jitter_m
            )
            # Perturb only grounded initial states; keep a quarter at the nominal start.
            apply = (t == 0) & (torch.rand(len(t), device=self.device) < 0.75)
            pose[:, :2] += (offset @ self.box_axes) * apply[:, None]
        elif any(self.cfg.box_offset_m):
            pose[:, :2] += (pose.new_tensor(self.cfg.box_offset_m) @ self.box_axes) * (t == 0)[:, None]
        velocity = torch.cat([self.box_lin_vel[t], self.box_ang_vel[t]], dim=1)
        velocity[t == 0] = 0
        self.box.write_root_state_to_sim(torch.cat([pose, velocity], -1), env_ids=env_ids)
        self.fresh[env_ids] = True
        self.full[env_ids] = t == 0
        if self.cfg.smooth_grasp:
            self.nonhand_contact[env_ids] = False
            self.nonhand_peak_n[env_ids] = 0
        for values in (self.lift, self.airborne, self.carried, self.success, self.stable):
            values[env_ids] = 0

    def _update_command(self):
        self.time_steps.add_((~self.fresh).long()).clamp_(max=self.motion.time_step_total - 1)
        self.fresh.zero_()
        self.update_relative_body_poses()

    def box_target(self, ticks):
        pos = self.box_pose[ticks, :3] + self._env.scene.env_origins
        if self.cfg.visual_pickup:
            blend = ((145 - ticks) / 45).clamp(0, 1)
            pos = pos + self.pickup_offset * (blend * blend * (3 - 2 * blend))[:, None]
        return pos

    def approach_offset(self, ticks):
        # Move the desired support position before closing the hands; blend back during carry.
        rise = (ticks / 42).clamp(0, 1)
        fall = ((145 - ticks) / 45).clamp(0, 1)
        w = (rise.square() * (3 - 2 * rise)) * (fall.square() * (3 - 2 * fall))
        offset = self.pickup_offset if ticks.ndim == 1 else self.pickup_offset[:, None]
        return 0.85 * w[..., None] * offset

    @property
    def anchor_pos_w(self):
        pos = super().anchor_pos_w
        return pos + self.approach_offset(self.time_steps) if self.cfg.visual_pickup else pos

    @property
    def body_pos_w(self):
        pos = super().body_pos_w
        return pos + self.approach_offset(self.time_steps)[:, None] if self.cfg.visual_pickup else pos

    @property
    def motion_anchor_pos_w_future(self):
        pos = super().motion_anchor_pos_w_future
        return pos + self.approach_offset(self._future_time_indices()) if self.cfg.visual_pickup else pos

    @property
    def anchor_lin_vel_w(self):
        vel = super().anchor_lin_vel_w
        if self.cfg.visual_pickup:
            vel = (
                vel
                + (self.approach_offset(self.time_steps + 1) - self.approach_offset(self.time_steps))
                / self._env.step_dt
            )
        return vel

    def observed_box_position(self):
        real = self.box.data.root_link_pos_w
        if self.cfg.visual_pickup:
            # Box rests on the floor before grasp; hand interaction supplies feedback later.
            visual = self.box_pose[0, :3] + self._env.scene.env_origins + self.pickup_offset
            w = self.full * ((self.first_grip + 10 - self.time_steps) / 10).clamp(0, 1)
            return torch.lerp(real, visual, w[:, None])
        return real

    def palm_local(self, position=None):
        d = self.box.data
        xyz = (
            self.robot.data.site_pos_w[:, self.sites]
            - (d.root_link_pos_w if position is None else position)[:, None, :]
        )
        return quat_apply_inverse(
            d.root_link_quat_w[:, None, :].expand(-1, 2, -1).reshape(-1, 4), xyz.reshape(-1, 3)
        ).reshape(-1, 2, 3)


@dataclass(kw_only=True)
class CarryCommandCfg(FutureMotionCommandCfg):
    grasp_feedback: bool = False
    full_body: bool = False
    smooth_grasp: bool = False
    visual_pickup: bool = False
    pickup_input: str = "normal"
    box_offset_m: tuple[float, float] = (0.0, 0.0)
    box_jitter_m: tuple[float, float] = (0.0, 0.0)

    def build(self, env):
        return CarryCommand(self, env)


def orientation6(quat):
    return matrix_from_quat(quat)[..., :2].flatten(-2)


def hoi_observation(env):
    c = motion(env)
    t = c.time_steps
    nxt = (t + 10).clamp(max=len(c.box_pose) - 1)
    d = c.box.data
    pelvis = c.robot_anchor_quat_w
    inv = quat_inv(pelvis)
    origin = c.robot_anchor_pos_w

    def pack(p, q):
        return torch.cat([quat_apply_inverse(pelvis, p - origin), orientation6(quat_mul(inv, q))], -1)

    observed = c.observed_box_position()
    real = pack(observed, d.root_link_quat_w)
    targets = [pack(c.box_target(k), c.box_pose[k, 3:]) for k in (t, nxt)]
    return torch.cat(
        [
            real,
            quat_apply_inverse(pelvis, d.root_link_lin_vel_w),
            quat_apply_inverse(pelvis, d.root_link_ang_vel_w),
            *targets,
            c.palms[t].flatten(1),
            c.grip[t],
            c.palm_local(observed).flatten(1),
        ],
        -1,
    )


def velocity_command(env):
    c = motion(env)
    v = quat_apply_inverse(c.anchor_quat_w, c.anchor_lin_vel_w)
    w = quat_apply_inverse(c.anchor_quat_w, c.anchor_ang_vel_w)
    z = torch.zeros_like(v[:, 0])
    return torch.stack((v[:, 0], v[:, 1], w[:, 2], z, z, z, z + 1, z), dim=1)


def end(env):
    c = motion(env)
    return c.time_steps >= len(c.box_pose) - 1


def dropped(env):
    c = motion(env)
    ref = c.box_target(c.time_steps)
    return (ref - c.box.data.root_link_pos_w).norm(dim=1) > 0.55


def interaction(env):
    c = motion(env)
    t = c.time_steps
    d = c.box.data
    xyz = d.root_link_pos_w - env.scene.env_origins
    quat = d.root_link_quat_w
    ref = c.box_pose[t]
    grip = c.grip[t]
    active = grip.min(dim=1).values
    pos_error = (d.root_link_pos_w - c.box_target(t)).norm(dim=-1)
    rot_error = 1 - (quat * ref[:, 3:]).sum(-1).square().clamp(0, 1)
    palm_error = (c.palm_local() - c.palms[t]).norm(dim=2)
    forces = env.scene["palms_box"].data.force.norm(dim=-1)
    floor = env.scene["box_floor"].data.force.norm(dim=-1).amax(dim=1)
    contact = forces > 0.5
    # Compute the hull bottom using the world-z row of the measured box rotation.
    bottom = xyz[:, 2] + (matrix_from_quat(quat)[:, 2] @ c.vertices.T).amin(dim=1)
    raised = (bottom > 0.15) & (floor < 0.5)
    c.lift |= raised & contact.all(dim=1)
    held = raised & contact.all(dim=1)
    c.air_start = torch.where((held & ~c.airborne)[:, None], xyz[:, :2], c.air_start)
    c.carried |= held & ((xyz[:, :2] - c.air_start).norm(dim=1) >= 0.75)
    c.airborne.copy_(held)
    terminal_phase = t >= len(c.box_pose) - 35
    upright = c.robot.data.projected_gravity_b[:, 2] < -0.75
    placed = (floor > 1.0) & ~contact.any(dim=1) & ((xyz[:, :2] - c.box_pose[-1, :2]).norm(dim=1) < 0.15)
    settled = (
        placed & (d.root_link_lin_vel_w.norm(dim=-1) < 0.05) & (d.root_link_ang_vel_w.norm(dim=-1) < 0.2)
    )
    c.stable.copy_(torch.where(terminal_phase & settled & upright, c.stable + 1, 0))
    c.success.copy_(end(env) & upright & c.carried & (c.stable >= 25))
    ended = env.termination_manager.dones & c.full
    for key in ("anchor_pos", "anchor_ori", "ee_body_pos", "dropped"):
        c.success &= ~env.termination_manager.get_term(key)
    c.counts[:4] += torch.stack(
        [ended.sum(), (ended & c.lift).sum(), (ended & c.carried).sum(), (ended & c.success).sum()]
    )
    if c.cfg.smooth_grasp:
        c.counts[4:] += torch.stack(
            [(ended & c.nonhand_contact).sum(), (ended & c.success & ~c.nonhand_contact).sum()]
        )
    if c.cfg.visual_pickup:
        far = ended & c.far_start
        c.far_counts += torch.stack(
            [
                far.sum(),
                (far & c.lift).sum(),
                (far & c.success).sum(),
                (far & c.success & ~c.nonhand_contact).sum(),
            ]
        )
    track = 3 * torch.exp(-pos_error.square() / 0.036) + torch.exp(-rot_error / 0.15)
    reach = (torch.exp(-palm_error.square() / 0.009) * (grip + 0.45)).mean(1)
    both = contact.float().mean(1) + contact.all(dim=1).float()
    slip = (d.root_link_lin_vel_w - c.box_lin_vel[t]).square().sum(dim=1)
    return (
        track
        + 2.8 * reach
        + active * (2.5 * both + 1.8 * raised.float() + torch.exp(-slip / 0.25))
        + terminal_phase * (placed.float() + settled.float())
    )


def set_hand_friction(spec):
    for s in ("left", "right"):
        g = spec.geom("robot/" + s + "_hand_collision")
        g.condim = 3
        g.priority = 1
        g.friction = [1.0, 0.005, 0.0001]
        g.contype = 1
        g.conaffinity = 1


def reference_pd(env):
    c = motion(env)
    action = env.action_manager.get_term("joint_pos")
    ids = action.target_ids
    if not hasattr(c, "reference_blend"):
        names = [c.robot.joint_names[i] for i in ids.tolist()]
        factors = [
            0.5 if "waist" in n else 0.65 if any(j in n for j in ("shoulder", "elbow", "wrist")) else 0.0
            for n in names
        ]
        assert factors.count(0.65) == 14 and factors.count(0.5) == 3
        if c.cfg.full_body:
            for i, name in enumerate(names):
                if "knee" in name:
                    factors[i] = 0.60
                elif "hip" in name or "ankle" in name:
                    factors[i] = 0.45
            assert factors.count(0.60) == 2 and factors.count(0.45) == 10 and min(factors) > 0
        c.reference_blend = torch.tensor(factors, device=env.device).expand(env.num_envs, -1)
    # Two control frames of preview compensate part of the finite PD response time.
    future = (c.time_steps + 2).clamp(max=c.motion.time_step_total - 1)
    joints = c.motion.joint_pos[future]
    if getattr(c.cfg, "grasp_feedback", False):
        from hoi_grasp_feedback import GraspFeedback

        if not hasattr(c, "grasp_controller"):
            c.grasp_controller = GraspFeedback(env, c)
            c.reference_blend = torch.where(c.reference_blend > 0.6, 0.92, c.reference_blend)
        joints = c.grasp_controller.target(c)
    target = (joints[:, ids] - action.offset) / action.scale
    return torch.cat((target, c.reference_blend), -1)


def grasp_tracking(env):
    c = motion(env)
    t = c.time_steps
    grip = c.grip[t].amin(dim=1)
    error = (c.palm_local() - c.palms[t]).norm(dim=-1).amax(dim=1)
    height_delta = c.box.data.root_link_pos_w[:, 2] - c.box_pose[t, 2] - env.scene.env_origins[:, 2]
    f = env.scene["palms_box"].data.force.norm(dim=-1).amin(dim=1)
    return grip * (
        2 * torch.exp(-error / 0.10) + (f / 22.0).clamp(0, 1) * (1 + torch.exp(-height_delta.abs() / 0.12))
    )


def upper_tracking(env):
    c = motion(env)
    if not hasattr(c, "upper_groups"):
        names = c.robot.joint_names
        c.upper_groups = [
            [i for i, n in enumerate(names) if any(g in n for g in group)]
            for group in (("shoulder", "elbow", "wrist"), ("waist",))
        ]
        c.torso_index = c.cfg.body_names.index("torso_link")
    arm, waist = c.upper_groups
    error = c.robot_joint_pos - c.joint_pos
    # Average per-joint kernels so a poorly tracked wrist does not erase elbow feedback.
    arm_score = torch.exp(-error[:, arm].square() / 0.30**2).mean(-1)
    waist_score = torch.exp(-error[:, waist].square() / 0.22**2).mean(-1)
    dot = (c.body_quat_w[:, c.torso_index] * c.robot_body_quat_w[:, c.torso_index]).sum(-1)
    torso_score = torch.exp(-(1 - dot.square().clamp(0, 1)) / 0.035)
    return 2 * arm_score + waist_score + 2 * torso_score


def lower_tracking(env):
    c = motion(env)
    if not hasattr(c, "leg_joint_ids"):
        c.leg_joint_ids = [
            i
            for i, name in enumerate(c.robot.joint_names)
            if any(k in name for k in ("hip", "knee", "ankle"))
        ]
        c.leg_body_ids = [
            c.cfg.body_names.index(f"{side}_{joint}_link")
            for side in ("left", "right")
            for joint in ("knee", "ankle_roll")
        ]
        assert len(c.leg_joint_ids) == 12
    angles = c.robot_joint_pos[:, c.leg_joint_ids] - c.joint_pos[:, c.leg_joint_ids]
    links = c.robot_body_pos_w[:, c.leg_body_ids] - c.body_pos_w[:, c.leg_body_ids]
    height = c.robot_anchor_pos_w[:, 2] - c.anchor_pos_w[:, 2]
    return (
        2 * torch.exp(-angles.square() / 0.30**2).mean(-1)
        + 2 * torch.exp(-links.square().sum(-1) / 0.12**2).mean(-1)
        + torch.exp(-height.square() / 0.09**2)
    )


def nonhand_box_force(env):
    c = motion(env)
    history = env.scene["nonhand_box"].data.force_history
    c.nonhand_peak_n = history.norm(dim=-1).amax(dim=(1, 2))
    c.nonhand_contact |= c.nonhand_peak_n > 5.0
    # Penalize a brush less than supporting a significant fraction of the 4 kg box.
    return ((c.nonhand_peak_n - 5.0).clamp_min(0) / 35.0).clamp_max(3)


def joint_tracking_excess(env):
    c = motion(env)
    angle = ((c.robot_joint_pos - c.joint_pos).abs() - 0.35).clamp_min(0) / 0.55
    speed = ((c.robot_joint_vel - c.joint_vel).abs() - 2.0).clamp_min(0) / 4.0
    # The worst joint must remain visible even if the other 28 follow the clip.
    return angle.amax(-1).clamp_max(4) + 0.5 * speed.amax(-1).clamp_max(4)


def target_curvature(env):
    a = env.action_manager
    # Second difference of physical PD targets in radians at 50 Hz.
    delta = (a.action - 2 * a.prev_action + a.prev_prev_action) * a.get_term("joint_pos").scale
    excess = ((delta.abs() - 0.10).clamp_min(0) / 0.25).clamp_max(4)
    return (excess.mean(-1) + 0.5 * excess.amax(-1)) * (env.episode_length_buf > 2)


def config(
    bundle,
    settings,
    n,
    seed,
    evaluate=False,
    reference_tracking=False,
    grasp_feedback=False,
    full_body=False,
    smooth_grasp=False,
    visual_pickup=False,
):
    if visual_pickup and not smooth_grasp:
        raise ValueError("Visual pickup requires smooth bimanual grasp")
    if full_body and (not reference_tracking or grasp_feedback):
        raise ValueError("Full-body tracking requires its own reference-conditioned controller")
    if smooth_grasp and not (reference_tracking and grasp_feedback):
        raise ValueError("Smooth grasp requires the two-hand compensation controller")
    cfg = sonic_tracking_env_cfg(str(bundle / "motion.npz"), il_ordered=False)
    cfg.seed = seed
    cfg.scene.num_envs = n
    cfg.scene.env_spacing = 0.0
    cfg.scene.entities["box"] = EntityCfg(spec_fn=partial(mujoco.MjSpec.from_file, str(bundle / "box.xml")))
    cfg.scene.spec_fn = set_hand_friction
    cfg.scene.sensors += (
        head_cam_cfg(),
        ContactSensorCfg(
            name="palms_box",
            primary=ContactMatch("geom", "(left|right)_hand_collision", "robot"),
            secondary=ContactMatch("geom", "object_collision", "box"),
            fields=("force",),
            reduce="netforce",
        ),
        ContactSensorCfg(
            name="box_floor",
            primary=ContactMatch("geom", "object_collision", "box"),
            secondary=ContactMatch("geom", "terrain"),
            fields=("force",),
            reduce="netforce",
        ),
    )
    old = cfg.commands["motion"]
    cmd = CarryCommandCfg(**vars(old))
    cmd.grasp_feedback = grasp_feedback
    cmd.full_body = full_body
    cmd.smooth_grasp = smooth_grasp
    cmd.visual_pickup = visual_pickup
    if smooth_grasp and not evaluate and not visual_pickup:
        cmd.box_jitter_m = (0.12, 0.10)
    cmd.sampling_mode = "start" if evaluate else "uniform"
    cmd.pose_range = {} if evaluate else {"x": (-0.01, 0.01), "y": (-0.01, 0.01)}
    cmd.velocity_range = {}
    cmd.joint_position_range = (0, 0) if evaluate else (-0.012, 0.012)
    cmd.debug_vis = False
    cfg.commands["motion"] = cmd
    cfg.events = {}
    cfg.sim.nconmax = 70
    cfg.sim.njmax = 300
    cfg.observations["policy"].enable_corruption = not evaluate
    for name, terms in [
        ("augmentation", robot_root_twist_cmd_terms({"command_name": "motion"})),
        ("q_proprio", proprio_terms()),
        ("velocity_command", {"cmd": ObservationTermCfg(func=velocity_command)}),
        ("hoi", {"cmd": ObservationTermCfg(func=hoi_observation)}),
    ]:
        cfg.observations[name] = ObservationGroupCfg(
            terms=terms, concatenate_terms=True, enable_corruption=False
        )
    cfg.observations["kv_tokens"] = ObservationGroupCfg(terms={}, concatenate_terms=False)
    configure_visual_env(cfg, settings)
    if visual_pickup:
        from hoi_pickup_vision import PickupVisualTokens

        visual = cfg.observations.pop("kv_tokens")
        visual.terms["img_tokens"].func = PickupVisualTokens
        cfg.observations = {"kv_tokens": visual, **cfg.observations}
    if reference_tracking:
        cfg.observations["hoi_reference_pd"] = ObservationGroupCfg(
            terms={"targets": ObservationTermCfg(func=reference_pd)},
            concatenate_terms=True,
            enable_corruption=False,
        )
        cfg.rewards["hoi_upper_tracking"] = RewardTermCfg(func=upper_tracking, weight=1.4)
    if full_body:
        cfg.rewards["hoi_lower_tracking"] = RewardTermCfg(func=lower_tracking, weight=1.4)
    if grasp_feedback:
        cfg.rewards["two_hand_grasp"] = RewardTermCfg(func=grasp_tracking, weight=1.5)
    if smooth_grasp:
        cfg.scene.sensors += (
            ContactSensorCfg(
                name="nonhand_box",
                primary=ContactMatch(
                    "geom", ".*_collision", "robot", exclude=("left_hand_collision", "right_hand_collision")
                ),
                secondary=ContactMatch("geom", "object_collision", "box"),
                fields=("force",),
                reduce="maxforce",
                history_length=cfg.decimation,
            ),
        )
        # Update the episode contact flag before interaction counts completed carries.
        cfg.rewards["nonhand_box_force"] = RewardTermCfg(func=nonhand_box_force, weight=-18.0)
        cfg.rewards["joint_tracking_excess"] = RewardTermCfg(func=joint_tracking_excess, weight=-3.0)
        cfg.rewards["target_curvature"] = RewardTermCfg(func=target_curvature, weight=-1.0)
    cfg.rewards["interaction"] = RewardTermCfg(func=interaction, weight=1.5)
    cfg.rewards["motion_global_root_pos"].weight = 2
    cfg.rewards["action_rate_l2"].weight = -0.025
    cfg.terminations["anchor_pos"].params["threshold"] = 0.45
    cfg.terminations["ee_body_pos"].params["threshold"] = 0.5
    cfg.terminations["anchor_ori"].params["threshold"] = 1.15
    cfg.terminations["clip_end"] = TerminationTermCfg(func=end)
    cfg.terminations["dropped"] = TerminationTermCfg(func=dropped)
    return cfg
