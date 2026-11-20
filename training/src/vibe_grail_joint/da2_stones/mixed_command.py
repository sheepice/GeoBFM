"""Pillar traversal plus independent flat standing, walking and continuous starts/stops."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

from ..kinematics import CHANNELS
from .command import StoneCommand, StoneCommandCfg


def task_roles(n, device):
    # Fixed proportions, even for an interleaved minibatch: 60/15/10/15.
    slot = torch.arange(n, device=device) % 20
    return torch.where(slot < 12, 0, torch.where(slot < 15, 1, torch.where(slot < 17, 2, 3)))


class MixedStoneCommand(StoneCommand):
    def _build_loader(self):
        motion = super()._build_loader()
        self._pool_end = motion.time_step_total
        self.flat_clip_start = motion.n_clips
        path = Path(self.cfg.flat_training['reference_bank'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.cfg.flat_training['reference_sha256']:
            raise ValueError('Flat reference bank identity changed')
        with np.load(path, allow_pickle=False) as bank:
            meta = json.loads(str(bank['metadata']))
            if meta['joint_names'] != list(self.robot.joint_names) or meta['body_names'] != list(self.cfg.body_names):
                raise ValueError('Flat reference joint/body ordering differs from simulation')
            if meta['dt'] != self._env.step_dt:
                raise ValueError('Flat reference control dt mismatch')
            if tuple(meta['names']) != ('stand', 'walk03', 'walk06', 'start_stop03', 'start_stop06', 'pillar_start06'):
                raise ValueError('Unknown flat reference schema')
            count, length = bank['command_speed'].shape
            self.reference_speeds = torch.as_tensor(bank['command_speed'].copy(), device=self.device)
            if getattr(self.cfg, 'standing_dagger', False):
                if (not meta.get('reference_only') or
                        meta.get('generated_by') != 'continuous SONIC context with explicit mode 0/2' or
                        self.reference_speeds[0].any()):
                    raise ValueError('Standing DAgger requires the original zero-speed SONIC stand stream')
            self.reference_length = length
            for name, tail in CHANNELS.items():
                values = torch.as_tensor(bank[name].copy(), device=self.device).reshape(-1, *tail)
                setattr(motion, name, torch.cat((getattr(motion, name), values)))
        offsets = self._pool_end + torch.arange(count, device=self.device) * length
        motion.clip_offsets = torch.cat((motion.clip_offsets, offsets))
        motion.clip_lengths = torch.cat((motion.clip_lengths, torch.full((count,), length, device=self.device)))
        motion.clip_ends = motion.clip_offsets + motion.clip_lengths
        motion.terrain_ids = torch.cat((motion.terrain_ids, torch.full((count,), -1, device=self.device)))
        motion.n_clips += count
        motion.time_step_total += count * length
        return motion

    def _upload_pool(self):
        for name, array in self.pool.channels().items():
            getattr(self.motion, name)[self.replay_frames:self._pool_end] = torch.as_tensor(array, device=self.device)

    def _init_task(self):
        super()._init_task()
        self.task_role = task_roles(self.num_envs, self.device)
        self.pillar_task = self.task_role == 0
        self.flat_task = ~self.pillar_task
        self.reference_id = torch.zeros_like(self.stream)
        self.reference_step = torch.zeros_like(self.stream)
        self.uses_clip = self.flat_task.clone()
        self.static_reset = self.flat_task.clone()
        self._last_mixed_step = int(self._env.common_step_counter)
        self.idle_anchor = torch.zeros(self.num_envs, 2, device=self.device)
        self.idle_steps = torch.zeros_like(self.stream)
        self.was_idle = torch.zeros_like(self.online)
        self.start_transitions = torch.zeros((), dtype=torch.long, device=self.device)
        self.stop_transitions = torch.zeros_like(self.start_transitions)
        # All worlds use the deployed online command flags, including flat worlds.
        # Training roles must not become a privileged mode bit in actor inputs.
        self.online[:] = True
        self.paired[:] = False
        self.flat[:] = False
        from .flat_dagger import role_mask
        selected = getattr(self.cfg, 'flat_dagger_roles', ()) or ((1,) if self.cfg.standing_dagger else ())
        self.dagger_mask[:] = role_mask(self.task_role, selected)
        self.ppo_mask[:] = ~self.dagger_mask
        self.ppo_online[:] = self.ppo_mask

    def dagger_strata(self):
        from .flat_dagger import reference_strata
        return reference_strata(self.reference_id[self.dagger_mask],
                                include_walk=2 in getattr(self.cfg, 'flat_dagger_roles', ()))

    def _sample_init_frame(self, env_ids):
        chosen, frames = super()._sample_init_frame(env_ids)
        role = self.task_role[env_ids]
        flat = role != 0
        static_pillar = (role == 0) & (torch.rand(len(env_ids), device=self.device) < self.cfg.flat_training['pillar_static_fraction'])
        use = flat | static_pillar
        refs = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        refs[role == 2] = torch.randint(1, 3, (int((role == 2).sum()),), device=self.device)
        refs[role == 3] = torch.randint(3, 5, (int((role == 3).sum()),), device=self.device)
        refs[static_pillar] = 5
        self.reference_id[env_ids] = refs
        self.reference_step[env_ids] = 0
        self.uses_clip[env_ids] = use
        self.static_reset[env_ids] = use & (role != 2)
        self.idle_steps[env_ids] = 0
        self.was_idle[env_ids] = False
        chosen[use] = self.flat_clip_start + refs[use]
        frames[use] = self.motion.clip_offsets[chosen[use]]
        self.phase[env_ids[use]] = 0
        self.speed[env_ids[use]] = self.reference_speeds[refs[use], 0]
        flat_tile_start = self.cfg.stone_levels * self.cfg.profile_variants
        self.tile[env_ids[flat]] = flat_tile_start + (env_ids[flat] % 2)
        ids = env_ids[use]
        spawn = self.tile_origins[self.tile[ids]].clone()
        spawn[:, 0] += torch.empty(len(ids), device=self.device).uniform_(*self.cfg.approach_spawn_x)
        spawn[:, 1] += torch.empty(len(ids), device=self.device).uniform_(-.04, .04)
        spawn[:, :2] -= self.motion.body_pos_w[frames[use], 0, :2]
        self._env.scene.env_origins[ids] = spawn
        return chosen, frames

    def _reset_task(self, env_ids, clip_ids, time_steps, origins):
        super()._reset_task(env_ids, clip_ids, time_steps, origins)
        ids = env_ids[self.static_reset[env_ids]]
        if len(ids):
            t = self.time_steps[ids]
            pos = self.motion.body_pos_w[t, 0].clone() + self._env.scene.env_origins[ids]
            ori = self.motion.body_quat_w[t, 0].clone()
            joints = self.motion.joint_pos[t].clone()
            # Half use the nominal deployment stand, half the generated stand.
            nominal = torch.rand(len(ids), device=self.device) < .5
            if getattr(self.cfg, 'standing_dagger', False):
                from .flat_dagger import role_mask
                selected = getattr(self.cfg, 'flat_dagger_roles', ()) or (1,)
                # DAgger roles start in their original SONIC reference stand.
                # Retain the previous initialization for PPO roles.
                nominal &= ~role_mask(self.task_role[ids], selected)
            pos[nominal, 2] = self.robot.data.default_root_state[ids[nominal], 2]
            ori[nominal] = self.robot.data.default_root_state[ids[nominal], 3:7]
            joints[nominal] = self.robot.data.default_joint_pos[ids[nominal]]
            joints += torch.empty_like(joints).uniform_(-.015, .015)
            limits = self.robot.data.soft_joint_pos_limits[ids]
            joints = joints.clamp(limits[..., 0], limits[..., 1])
            yaw = torch.empty(len(ids), device=self.device).uniform_(-.05, .05)
            ori = quat_mul(quat_from_euler_xyz(yaw * 0, yaw * 0, yaw), ori)
            self._write_reference_state_to_sim(ids, pos, ori, torch.zeros_like(pos),
                                               torch.zeros_like(pos), joints, torch.zeros_like(joints))
        self.idle_anchor[env_ids] = self.robot.data.root_link_pos_w[env_ids, :2]

    def _update_command(self, env_ids=None):
        step = int(self._env.common_step_counter)
        new_step = env_ids is None and step != self._last_mixed_step
        if new_step:
            self._last_mixed_step = step
            changed, recenter = self.pool.advance()
            ids = (~self.uses_clip).nonzero().flatten()
            self._env.scene.env_origins[ids, :2] += torch.as_tensor(recenter, device=self.device)[self.stream[ids]]
            if changed:
                self._upload_pool()
            # Resets in this physics step have episode length 0: leave frame 0 intact.
            self.reference_step += (self.uses_clip & (self._env.episode_length_buf > 0)).long()
            self.reference_step.clamp_max_(self.reference_length - 51)
        dynamic = (~self.uses_clip).nonzero().flatten()
        self.time_steps[dynamic] = self.replay_frames + self.stream[dynamic] * self.pool.capacity + self.pool.cursor - self.phase[dynamic]
        self.speed[dynamic] = self.cfg.fixed_speed
        ids = self.uses_clip.nonzero().flatten()
        ref, frame = self.reference_id[ids], self.reference_step[ids]
        self.time_steps[ids] = self.motion.clip_offsets[self.flat_clip_start + ref] + frame
        self.speed[ids] = self.reference_speeds[ref, frame]
        idle = self.speed < .001
        if new_step:
            live = self.flat_task & (self._env.episode_length_buf > 1)
            self.start_transitions += (live & self.was_idle & ~idle).sum()
            self.stop_transitions += (live & ~self.was_idle & idle).sum()
        if env_ids is not None or new_step:
            selected = self.online_ids if env_ids is None else env_ids
            entered = selected[idle[selected] & ~self.was_idle[selected]]
            self.idle_anchor[entered] = self.robot.data.root_link_pos_w[entered, :2]
            self.idle_steps[selected] = torch.where(idle[selected], self.idle_steps[selected] + int(new_step), 0)
            settled = selected[idle[selected] & (self.idle_steps[selected] == 40)]
            self.idle_anchor[settled] = self.robot.data.root_link_pos_w[settled, :2]
            self.was_idle[selected] = idle[selected]
        self.update_relative_body_poses()


@dataclass(kw_only=True)
class MixedStoneCommandCfg(StoneCommandCfg):
    flat_training: dict | None = None
    standing_dagger: bool = False
    flat_dagger_roles: tuple[int, ...] = ()

    def build(self, env):
        return MixedStoneCommand(self, env)
