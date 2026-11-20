"""Per-world planner/upper-motion composition without modifying shared motion banks."""

from dataclasses import dataclass
import torch
from vibe_grail_joint.velocity.dagger_task import DaggerMotionCommand, DaggerMotionCommandCfg
from .fk import RobotFK
from .contracts import mul, rotate, points_from_bodies, inv


class VRMotionCommand(DaggerMotionCommand):
    def _init_task(self):
        super()._init_task()
        self.vr_fk = RobotFK(self.device)
        if list(self.robot.joint_names) != self.vr_fk.joint_names:
            raise ValueError("FK joint order mismatch")
        self.vr_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.vr_clip = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.vr_phase = torch.zeros_like(self.vr_clip)
        self.vr_started = torch.zeros_like(self.vr_clip)
        self._vr_cache = None
        self._vr_external = None
        self._vr_external_mask = None

    def _sample_init_frame(self, env_ids):
        chosen, frames = super()._sample_init_frame(env_ids)
        self._sample_upper(env_ids)
        return chosen, frames

    def _sample_upper(self, env_ids):
        ids = env_ids[self.online[env_ids]]
        if len(ids):
            sources = [self.source_groups[n] for n in self.cfg.vr_upper_sources]
            candidates = torch.cat(sources)
            self.vr_clip[ids] = candidates[
                torch.randint(len(candidates), (len(ids),), device=self.device)
            ]
            lengths = self.motion.clip_lengths[self.vr_clip[ids]]
            self.vr_phase[ids] = (torch.rand(len(ids), device=self.device) * lengths).long()
            self.vr_active[ids] = (
                torch.rand(len(ids), device=self.device) < self.cfg.vr_upper_probability
            )
            self.vr_started[ids] = int(self._env.common_step_counter)
        self._vr_cache = None

    def dagger_strata(self):
        original = super().dagger_strata()
        ids = self.dagger_mask.nonzero().flatten()
        augmented = self.dagger_planner[ids] & self.vr_active[ids]
        return original + augmented.long() * len(self.cfg.planner_speeds)

    def _update_command(self, env_ids=None):
        self._vr_cache = None
        super()._update_command(env_ids)

    def _composite(self, next_step=0):
        steps = self.time_steps + next_step
        ends = self.motion.clip_ends[self._clip_ids] - 1
        # Online rolling planner always has 0.9 s of valid future reference.
        steps = torch.minimum(steps, ends)
        result = self.motion.joint_pos[steps].clone()
        elapsed = (int(self._env.common_step_counter) - self.vr_started + next_step).clamp_min(0)
        lengths = self.motion.clip_lengths[self.vr_clip].clamp_min(2)
        period = 2 * (lengths - 1)
        phase = (self.vr_phase + elapsed) % period
        local = torch.where(phase < lengths, phase, period - phase)
        target = self.motion.joint_pos[self.motion.clip_offsets[self.vr_clip] + local]
        ramp = (elapsed.float() * self._env.step_dt / self.cfg.vr_blend_seconds).clamp(0, 1)
        weight = (self.online & self.vr_active).float() * ramp
        result[:, 12:] += weight[:, None] * (target[:, 12:] - result[:, 12:])
        return result

    def _refresh(self):
        if self._vr_cache is not None:
            return self._vr_cache
        joints = self._composite()
        next_joints = self._composite(1)
        velocity = (next_joints - joints) / self._env.step_dt
        velocity = torch.where(
            self.online[:, None], velocity, self.motion.joint_vel[self.time_steps]
        )
        # Recompute both frames in one batched FK call. This also updates reference
        # body velocities; no stale pre-composition hand/torso velocity enters critic.
        local_p, local_q = self.vr_fk(torch.stack((joints, next_joints), 1))
        next_steps = torch.minimum(self.time_steps + 1, self.motion.clip_ends[self._clip_ids] - 1)
        index = torch.stack((self.time_steps, next_steps), 1)
        root_p = self.motion.body_pos_w[index, 0] + self._env.scene.env_origins[:, None, :]
        root_q = self.motion.body_quat_w[index, 0]
        world_p = rotate(root_q[:, :, None], local_p) + root_p[:, :, None]
        world_q = mul(root_q[:, :, None], local_q)
        body_pos = torch.where(
            self.online[:, None, None],
            world_p[:, 0],
            self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None],
        )
        body_quat = torch.where(
            self.online[:, None, None], world_q[:, 0], self.motion.body_quat_w[self.time_steps]
        )
        linear = (world_p[:, 1] - world_p[:, 0]) / self._env.step_dt
        delta = mul(world_q[:, 1], inv(world_q[:, 0]))
        delta = torch.where(delta[..., :1] < 0, -delta, delta)
        magnitude = delta[..., 1:].norm(dim=-1, keepdim=True)
        angular = (
            delta[..., 1:]
            * (2 * torch.atan2(magnitude, delta[..., :1]) / magnitude.clamp_min(1e-8))
            / self._env.step_dt
        )
        linear = torch.where(
            self.online[:, None, None], linear, self.motion.body_lin_vel_w[self.time_steps]
        )
        angular = torch.where(
            self.online[:, None, None], angular, self.motion.body_ang_vel_w[self.time_steps]
        )
        points, orientations = points_from_bodies(
            body_pos, body_quat, self.cfg.body_names, root_p[:, 0], root_q[:, 0]
        )
        if self._vr_external is not None:
            p, q = self._vr_external
            points = torch.where(self._vr_external_mask[:, None, None], p, points)
            orientations = torch.where(self._vr_external_mask[:, None, None], q, orientations)
        self._vr_cache = dict(
            joints=joints,
            velocity=velocity,
            body_pos=body_pos,
            body_quat=body_quat,
            body_lin_vel=linear,
            body_ang_vel=angular,
            points=points,
            orientations=orientations,
        )
        return self._vr_cache

    @property
    def body_lin_vel_w(self):
        return self._refresh()["body_lin_vel"]

    @property
    def body_ang_vel_w(self):
        return self._refresh()["body_ang_vel"]

    @property
    def joint_pos(self):
        return self._refresh()["joints"]

    @property
    def joint_vel(self):
        return self._refresh()["velocity"]

    @property
    def body_pos_w(self):
        return self._refresh()["body_pos"]

    @property
    def body_quat_w(self):
        return self._refresh()["body_quat"]

    @property
    def vr_points(self):
        return self._refresh()["points"]

    @property
    def vr_orientations(self):
        return self._refresh()["orientations"]

    def set_external_targets(self, positions=None, orientations=None, mask=None):
        """Root-local positions and root-local wxyz orientations, L/R/head order.

        None immediately restores planner defaults; live deployments should use TargetRouter
        for a continuous transition before calling this method.
        """
        if positions is None:
            self._vr_external = None
            self._vr_external_mask = None
        else:
            if positions.shape != (self.num_envs, 3, 3) or orientations.shape != (
                self.num_envs,
                3,
                4,
            ):
                raise ValueError("Invalid external target shape")
            if not torch.isfinite(positions).all() or not torch.isfinite(orientations).all():
                raise ValueError("Nonfinite external target")
            if (orientations.norm(dim=-1) < 1e-6).any():
                raise ValueError("Zero target quaternion")
            self._vr_external = (positions, orientations / orientations.norm(dim=-1, keepdim=True))
            self._vr_external_mask = (
                self.online.clone()
                if mask is None
                else mask.to(device=self.device, dtype=torch.bool) & self.online
            )
        self._vr_cache = None


@dataclass(kw_only=True)
class VRMotionCommandCfg(DaggerMotionCommandCfg):
    vr_upper_probability: float = 0.5
    vr_upper_sources: tuple = ("sonic",)
    vr_blend_seconds: float = 0.5

    def build(self, env):
        return VRMotionCommand(self, env)
