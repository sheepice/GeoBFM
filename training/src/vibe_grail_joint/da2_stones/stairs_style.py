"""Stones PPO with the original stairs planner/replay DAgger and safe-flat prior.

Opt-in recipe: preserve 60% stones PPO; split 40% flat equally between the
stairs online-planner and original motion-source DAgger. No timed start clips.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch

from ..kinematics import CHANNELS
from ..velocity.command import VelocityMotionCommand, sample_flat_clips, sample_planner_streams
from ..velocity.dagger_task import DaggerMotionCommand
from ..velocity.dagger_env import DaggerVecEnv
from .command import StoneCommand, StoneCommandCfg
from . import turning


def enabled(settings):
    return settings.get('stones_stairs_style') is not None


def validate_settings(settings):
    cfg = settings.get('stones_stairs_style')
    if cfg is None:
        return
    if cfg != dict(version=1, fractions=dict(pillar_ppo=.6, flat_planner_dagger=.2, flat_replay_dagger=.2)):
        raise ValueError('Stairs-style stones requires the 60/20/20 mixture')
    if any(settings.get(k) is not None for k in ('flat_training', 'flat_dagger', 'flat_standing_dagger')):
        raise ValueError('Stairs-style stones cannot mix legacy timed flat references')
    dagger = settings.get('dagger')
    if not dagger or dagger['teacher_checkpoint'] != settings['sonic_checkpoint']:
        raise ValueError('Stairs-style DAgger requires the original frozen SONIC teacher')
    for key in ('coefficient', 'batch_size', 'buffer_capacity'):
        if not math.isfinite(dagger[key]) or dagger[key] <= 0:
            raise ValueError('Invalid stairs DAgger setting: ' + key)
    speeds = settings['planner_speeds']
    if tuple(speeds) != (0., settings['stones']['speed_m_s']):
        raise ValueError('Keep the requested zero / nominal stone speed streams')
    if not settings.get('bank_root'):
        raise ValueError('Original flat replay requires the complete per-rank bank_root')
    prior = settings['flat_repair']
    if not 0 <= prior['zero_edge_distance_m'] < prior['full_edge_distance_m']:
        raise ValueError('Invalid safe-flat edge distances')
    if prior['startup_seconds'] != 1. or prior['startup_multiplier'] != 2.:
        raise ValueError('Use the original stairs startup prior')
    turning.validate_settings(settings.get('flat_turning'))


def task_masks(n, device):
    if n < 20 or n % 20:
        raise ValueError('Stairs-style stones requires num_envs divisible by 20')
    slot = torch.arange(n, device=device) % 20
    return slot < 12, (slot >= 12) & (slot < 16), slot >= 16


class StairsStyleStoneCommand(StoneCommand):
    def _build_loader(self):
        motion = super()._build_loader()
        self._pool_end = motion.time_step_total
        self._turn_clip_start = motion.n_clips
        if self.cfg.flat_turning is None:
            return motion
        seed_pose = np.concatenate((self.robot.data.default_root_state[0, :7].cpu().numpy(),
                                   self.robot.data.default_joint_pos[0].cpu().numpy())).astype(np.float32)
        seed_pose[:2], seed_pose[3:7] = 0., [1., 0., 0., 0.]
        channels, headings, yaw_rates = turning.generate_references(
            self.cfg.planner_path, seed_pose, self.cfg.flat_turning['yaw_rate_rad_s'],
            self._env.step_dt, self.cfg.seed)
        count, length = headings.shape
        for name, tail in CHANNELS.items():
            values = torch.as_tensor(channels[name].reshape(-1, *tail), device=self.device)
            setattr(motion, name, torch.cat((getattr(motion, name), values)))
        offsets = self._pool_end + torch.arange(count, device=self.device) * length
        motion.clip_offsets = torch.cat((motion.clip_offsets, offsets))
        motion.clip_lengths = torch.cat((motion.clip_lengths, torch.full_like(offsets, length)))
        motion.clip_ends = torch.cat((motion.clip_ends, offsets + length))
        motion.terrain_ids = torch.cat((motion.terrain_ids, torch.full_like(offsets, -1)))
        motion.n_clips += count
        motion.time_step_total += count * length
        self._turn_headings = torch.as_tensor(headings, device=self.device)
        self._turn_rates = torch.as_tensor(yaw_rates, device=self.device)
        return motion

    def _upload_pool(self):
        for name, array in self.pool.channels().items():
            getattr(self.motion, name)[self.replay_frames:self._pool_end] = torch.as_tensor(array, device=self.device)

    def _init_task(self):
        super()._init_task()
        self.pillar_task, self.dagger_planner, self.flat = task_masks(self.num_envs, self.device)
        self.ppo_online = self.pillar_task.clone()
        self.ppo_mask = self.pillar_task.clone()
        self.dagger_mask = self.dagger_planner | self.flat
        self.flat_task = self.dagger_mask.clone()
        self.paired = torch.zeros_like(self.pillar_task)
        # Exactly the stairs actor modes: planner is online, replay is flat.
        self.online = self.pillar_task | self.dagger_planner
        self.online_ids = self.online.nonzero().flatten()
        self.task_role = torch.where(self.pillar_task, 0, torch.where(self.dagger_planner, 1, 2))
        self.metrics['terrain_fraction'] = self.pillar_task.float()
        self._source_index = torch.zeros(self.motion.n_clips, device=self.device, dtype=torch.long)
        for i, source in enumerate(self.flat_sources):
            self._source_index[self.source_groups[source]] = i
        self.idle_steps = torch.zeros_like(self.stream)
        self.idle_anchor = torch.zeros(self.num_envs, 2, device=self.device)
        self.was_idle = torch.zeros_like(self.online)
        self._last_idle_step = int(self._env.common_step_counter)
        if getattr(self.cfg, 'flat_turning', None):
            self.turn_role = turning.flat_turn_roles(self.dagger_planner, self.cfg.flat_turning['fraction_of_planner'])
            self.turn_task = self.turn_role != 0
            self._turn_ids = self.turn_task.nonzero().flatten()
            self.online_ids = (self.online & ~self.turn_task).nonzero().flatten()
            n_straight = int(self.dagger_planner.sum()) - self._turn_ids.numel()
            self._stand_probability = turning.STAND_FRACTION * (n_straight + self._turn_ids.numel()) / n_straight
            self._turn_initial_frame = torch.zeros_like(self.stream)
            self._turn_clip = torch.zeros_like(self.stream)
            self.turn_yaw_rate = self.speed.new_zeros(self.num_envs)
            self.turn_heading = self.speed.new_zeros(self.num_envs)

    def dagger_strata(self):
        strata = DaggerMotionCommand.dagger_strata(self)
        if hasattr(self, 'turn_task'):
            base = len(self.flat_sources) + len(self.cfg.planner_speeds)
            roles = self.turn_role[self.dagger_mask]
            strata = torch.where(roles != 0, base + (roles < 0).long(), strata)
        return strata

    def _sample_init_frame(self, env_ids):
        # Keep the stone curriculum, layout sampling and randomized approach.
        chosen, frames = StoneCommand._sample_init_frame(self, env_ids)
        flat_ids = env_ids[self.flat_task[env_ids]]
        flat_start = getattr(self, 'flat_tile_start', self.cfg.stone_levels * self.cfg.profile_variants)
        self.tile[flat_ids] = flat_start + flat_ids % 2
        mask = self.online[env_ids]
        if hasattr(self, 'turn_task'):
            mask = mask & ~self.turn_task[env_ids]
        ids = env_ids[mask]
        self.stream[ids] = sample_planner_streams(self.cfg.planner_speeds, len(ids), self.device)
        if hasattr(self, 'turn_task'):
            # Retain the original 6% unconditional stand probability among all
            # planner DAgger environments. The new turn slots come from walking.
            straight = self.dagger_planner[ids]
            planner_ids = ids[straight]
            self.stream[planner_ids] = (torch.rand_like(self.speed[planner_ids]) >= self._stand_probability).long()
        self.phase[ids] = torch.randint(0, 20, (len(ids),), device=self.device)
        chosen[mask] = self.replay_clips + self.stream[ids]
        frames[mask] = self.motion.clip_offsets[chosen[mask]] + self.pool.cursor - self.phase[ids]
        nominal = torch.tensor(self.cfg.planner_speeds, device=self.device)[self.stream[ids]]
        self.speed[ids] = torch.where(nominal > 0,
            (nominal + torch.empty_like(nominal).uniform_(-.04, .04)).clamp(min=.15), 0.)
        if hasattr(self, 'turn_task'):
            local = self.turn_task[env_ids]
            ids = env_ids[local]
            count = len(turning.TURN_DURATIONS)
            clips = torch.randint(count, (len(ids),), device=self.device) + count * (self.turn_role[ids] < 0).long()
            self._turn_clip[ids] = clips
            self._turn_initial_frame[ids] = torch.randint(0, 11, (len(ids),), device=self.device)
            chosen[local] = self._turn_clip_start + clips
            frames[local] = self.motion.clip_offsets[chosen[local]] + self._turn_initial_frame[ids]
            self.stream[ids] = 0
            self.speed[ids] = 0
            self.turn_yaw_rate[ids] = 0
            self.turn_heading[ids] = 0
        replay_mask = self.flat[env_ids]
        replay_ids = env_ids[replay_mask]
        if len(replay_ids):
            clips = sample_flat_clips(self, len(replay_ids))
            chosen[replay_mask] = clips
            lengths = (self.motion.clip_lengths[clips].float() * self._init_phase_max).long().clamp(min=1)
            local = (torch.rand(len(clips), device=self.device) * lengths).long()
            if self.cfg.start_from_zero:
                local.zero_()
            frames[replay_mask] = self.motion.clip_offsets[clips] + local
            self.speed[replay_ids] = self.motion.body_lin_vel_w[frames[replay_mask], 0, 0]
        # Both original replay and online planner flat tasks live on actual flat
        # stone-atlas tiles, never on the original bank's paired stair origins.
        spawn = self.tile_origins[self.tile[env_ids]].clone()
        x_perturbation = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.approach_spawn_x)
        if hasattr(self, 'approach_x_scale'):
            x_perturbation *= self.approach_x_scale[self.tile[env_ids]]
        spawn[:, 0] += x_perturbation
        spawn[:, 1] += torch.empty(len(env_ids), device=self.device).uniform_(-.04, .04)
        spawn[:, :2] -= self.motion.body_pos_w[frames, 0, :2]
        self._env.scene.env_origins[env_ids] = spawn
        self.idle_steps[env_ids] = 0
        self.was_idle[env_ids] = False
        return chosen, frames

    def _reset_task(self, env_ids, clip_ids, time_steps, origins):
        super()._reset_task(env_ids, clip_ids, time_steps, origins)
        self.idle_anchor[env_ids] = self.robot.data.root_link_pos_w[env_ids, :2]

    def _update_specialized_reference(self, env_ids):
        if not hasattr(self, 'turn_task'):
            return
        ids = self._turn_ids if env_ids is None else env_ids[self.turn_task[env_ids]]
        clips = self._turn_clip[ids]
        frame = (self._env.episode_length_buf[ids] + self._turn_initial_frame[ids]).clamp_max(
            self._turn_headings.shape[1] - 46)
        self.time_steps[ids] = self.motion.clip_offsets[self._clip_ids[ids]] + frame
        self.turn_yaw_rate[ids] = self._turn_rates[clips, frame]
        self.turn_heading[ids] = self._turn_headings[clips, frame]

    def _update_command(self, env_ids=None):
        # Important: bypass StoneCommand's fixed-speed fill. This is the SAME
        # rolling planner/replay update as stairs: idle stays idle indefinitely.
        VelocityMotionCommand._update_command(self, env_ids)
        replay = (~self.online).nonzero().flatten()
        self.speed[replay] = self.motion.body_lin_vel_w[self.time_steps[replay], 0, 0]
        step = int(self._env.common_step_counter)
        advance = env_ids is None and step != self._last_idle_step
        if advance:
            self._last_idle_step = step
        if env_ids is not None or advance:
            ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
            idle = self.online[ids] & (self.speed[ids] < .001)
            if hasattr(self, 'turn_task'):
                idle &= ~self.turn_task[ids] | (self.turn_yaw_rate[ids].abs() < .001)
            entered = ids[idle & ~self.was_idle[ids]]
            self.idle_anchor[entered] = self.robot.data.root_link_pos_w[entered, :2]
            self.idle_steps[ids] = torch.where(idle, self.idle_steps[ids] + int(advance), 0)
            settled = ids[idle & (self.idle_steps[ids] == 40)]
            self.idle_anchor[settled] = self.robot.data.root_link_pos_w[settled, :2]
            self.was_idle[ids] = idle


@dataclass(kw_only=True)
class StairsStyleStoneCommandCfg(StoneCommandCfg):
    flat_turning: dict | None = None

    def build(self, env):
        return StairsStyleStoneCommand(self, env)


@torch.no_grad()
def flat_supervision(env, settings):
    """Stairs prior with circular edge geometry instead of stair X edges.

    A conservative disk bounds each entire foot's horizontal extent. Reject
    non-flat physical neighborhoods (including mats) with the stairs check.
    """
    c = env.command_manager.get_term('motion')
    feet = c.robot_body_pos_w[:, [3, 6]]
    target = c.primitives.values[c.tile]
    centers = target[:, :, :2] + c.tile_origins[c.tile, None, :2]
    distance = (feet[:, :, None, :2] - centers[:, None]).norm(dim=-1)
    distance -= target[:, None, :, 2] + math.hypot(.17, .08)
    distance = torch.where(target[:, None, :, 2] > 0, distance, torch.inf).flatten(1).amin(-1)
    weight = ((distance - settings['zero_edge_distance_m']) /
              (settings['full_edge_distance_m'] - settings['zero_edge_distance_m'])).clamp(0, 1)
    offsets = feet.new_tensor([[-.17, -.08], [-.17, .08], [.17, -.08], [.17, .08]])
    points = (feet[:, :, None, :2] + offsets).flatten(1, 2)
    heights = c.ground_height(points)
    flat = heights.amax(-1) - heights.amin(-1) <= settings['height_tolerance_m']
    origin = c.tile_origins[c.tile]
    inside = ((feet[:, :, 1] - origin[:, None, 1]).abs() < 1.65).all(-1)
    weight *= (flat & inside & c.ppo_online).float()
    startup = env.episode_length_buf * env.step_dt < settings['startup_seconds']
    weight *= torch.where(startup, settings['startup_multiplier'], 1.)
    return weight[:, None]


class StairsStyleDaggerEnv(DaggerVecEnv):
    """Original stairs collection, teacher, buffer and loss; stone sidecar format."""
    def __init__(self, wrapped, settings, rank):
        super().__init__(wrapped, settings, rank)
        if settings.get('flat_turning'):
            from ..velocity.dagger_buffer import DaggerBuffer

            existing = self.buffer.strata
            # Divide the old walking group among straight, left and right,
            # retaining the original pure-stand and motion-bank weights.
            groups = (*range(existing), existing - 1, existing - 1)
            self.buffer = DaggerBuffer(self.buffer.capacity, len(groups), settings['seed'] + rank,
                                       sampling_groups=groups)

    def state_dict(self):
        return dict(version=1, teacher_sha256=self.teacher_sha256,
                    num_worlds=self.unwrapped.num_envs, replay=self.buffer.state_dict(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(self.device))

    def load_state_dict(self, state):
        if (state['version'], state['teacher_sha256'], state['num_worlds']) != (
                1, self.teacher_sha256, self.unwrapped.num_envs):
            raise ValueError('Stairs-style DAgger replay/teacher/world-count mismatch')
        self.buffer.load_state_dict(state['replay'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'], self.device)


def task_audit(c):
    ppo, planner, replay = task_masks(c.num_envs, c.device)
    if (not torch.equal(c.ppo_mask, ppo) or not torch.equal(c.ppo_online, ppo)
            or not torch.equal(c.dagger_planner, planner) or not torch.equal(c.flat, replay)
            or not torch.equal(c.dagger_mask, planner | replay) or c.paired.any()
            or not torch.equal(c.online, ppo | planner)):
        raise RuntimeError('Stairs-style stone task allocation changed')
    if not (c.tile_data['kind'][c.tile[planner | replay]] == 3).all():
        raise RuntimeError('Flat DAgger left its physical flat tiles')
    if not torch.isin(c._clip_ids[replay], torch.cat([c.source_groups[s] for s in c.flat_sources])).all():
        raise RuntimeError('Flat DAgger selected a non-original-flat replay clip')
    turns = getattr(c, 'turn_task', torch.zeros_like(c.online))
    rolling = c.online & ~turns
    if not torch.equal(c._clip_ids[rolling], c.replay_clips + c.stream[rolling]):
        raise RuntimeError('Online stone/planner worlds used a timed or replay reference')
    if (turns.any() and (not c.dagger_planner[turns].all() or (c.speed[turns] != 0).any()
                        or (c._clip_ids[turns] < c._turn_clip_start).any())):
        raise RuntimeError('Flat turn DAgger reference or zero translation is incorrect')
    idle = ppo & (c.stream == 0)
    if not (c.speed[idle] == 0).all():
        raise RuntimeError('Zero-speed stone commands were overwritten')
    stats = dict(pillar_online_ppo=int(ppo.sum()), flat_planner_dagger=int(planner.sum()),
                flat_replay_dagger=int(replay.sum()), paired_stairs=0,
                obstacle_idle_worlds=int(idle.sum()),
                obstacle_stream_counts=torch.bincount(c.stream[ppo], minlength=len(c.cfg.planner_speeds)).tolist(),
                flat_source_weights={s:float(c.flat_weights[i]) for i,s in enumerate(c.flat_sources)},
                flat_source_counts={s:int(torch.isin(c._clip_ids[replay], c.source_groups[s]).sum()) for s in c.flat_sources})
    if hasattr(c, 'turn_task'):
        stats.update(flat_turn_left=int((c.turn_role == 1).sum()), flat_turn_right=int((c.turn_role == -1).sum()),
                     flat_turn_active=int((c.turn_yaw_rate != 0).sum()))
    return stats
