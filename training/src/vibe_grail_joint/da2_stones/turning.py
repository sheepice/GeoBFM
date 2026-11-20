"""Flat DAgger references from SONIC's native zero-translation turning mode."""

import math

import numpy as np
import torch

from ..kinematics import angular_velocity
from ..velocity.planner import PlannerPool


EPISODE_SECONDS = 7.
TURN_START_SECONDS = .7
TURN_DURATIONS = (1.6, 2.4, 3.2)
STAND_FRACTION = .06


def validate_settings(settings):
    if settings is None:
        return
    if (set(settings) != {'version', 'fraction_of_planner', 'yaw_rate_rad_s'}
            or settings['version'] != 1
            or not 0. < settings['fraction_of_planner'] < 1. - STAND_FRACTION
            or not 0. < settings['yaw_rate_rad_s'] <= .5):
        raise ValueError('Invalid flat turn reference settings')


def flat_turn_timeout(env):
    command = env.command_manager.get_term('motion')
    return command.turn_task & (env.episode_length_buf >= round(EPISODE_SECONDS / env.step_dt))


def flat_turn_roles(planner_mask, fraction):
    """Reserve equal left/right slots within online flat DAgger worlds."""
    planner_ids = planner_mask.nonzero().flatten()
    per_side = max(1, round(len(planner_ids) * fraction / 2))
    if 2 * per_side >= len(planner_ids):
        raise ValueError('Flat DAgger needs room for stand, walk and both turn directions')
    roles = torch.zeros_like(planner_mask, dtype=torch.long)
    roles[planner_ids[:per_side]] = 1
    roles[planner_ids[per_side:2 * per_side]] = -1
    return roles


class _TurnPool(PlannerPool):
    def __init__(self, path, seed_pose, count, seed):
        self.heading = np.zeros(count, dtype=np.float64)
        super().__init__(path, seed_pose, [0.] * count, seed)

    def _directions(self, stream):
        yaw = self.heading[stream]
        facing = np.array([[np.cos(yaw), np.sin(yaw), 0.]], dtype=np.float32)
        return np.zeros_like(facing), facing


def generate_references(path, seed_pose, yaw_rate, dt, seed):
    """Generate start, turn, and settled hold using the same rolling context."""
    if not math.isclose(dt, .02, abs_tol=1e-8):
        raise ValueError('Flat turn references require 50 Hz control')
    durations = np.tile(TURN_DURATIONS, 2)
    rates = np.repeat([yaw_rate, -yaw_rate], len(TURN_DURATIONS))
    pool = _TurnPool(path, seed_pose, len(rates), seed)
    # Include the ten-frame SONIC lookahead after the one-turn episode.
    frames = round(EPISODE_SECONDS / dt) + 51
    poses = np.empty((len(rates), frames, len(seed_pose)), dtype=np.float32)
    headings = np.empty((len(rates), frames), dtype=np.float32)
    yaw_rates = np.empty_like(headings)
    translation = np.zeros((len(rates), 2), dtype=np.float32)
    for step in range(frames):
        elapsed = step * dt - TURN_START_SECONDS
        pool.heading[:] = rates * np.clip(elapsed, 0., durations)
        headings[:, step] = pool.heading
        yaw_rates[:, step] = rates * ((elapsed >= -1e-8) & (elapsed < durations - 1e-8))
        if step:
            _, shift = pool.advance()
            translation += shift
        poses[:, step] = pool.poses[:, pool.cursor]
        poses[:, step, :2] += translation
    poses[:, :, :2] -= poses[:, :1, :2].copy()
    n_bodies = len(pool.body_ids)
    pos = np.empty((len(rates), frames, n_bodies, 3), np.float32)
    quat = np.empty((len(rates), frames, n_bodies, 4), np.float32)
    import mujoco

    for clip, trajectory in enumerate(poses):
        for frame, qpos in enumerate(trajectory):
            pool.data.qpos[:] = qpos
            mujoco.mj_kinematics(pool.model, pool.data)
            pos[clip, frame] = pool.data.xpos[pool.body_ids]
            quat[clip, frame] = pool.data.xquat[pool.body_ids]
    channels = {
        'joint_pos': poses[:, :, 7:], 'joint_vel': np.gradient(poses[:, :, 7:], dt, axis=1),
        'body_pos_w': pos, 'body_quat_w': quat, 'body_lin_vel_w': np.gradient(pos, dt, axis=1),
        'body_ang_vel_w': np.stack([angular_velocity(clip, dt) for clip in quat]),
    }
    if not all(np.isfinite(array).all() for array in channels.values()):
        raise RuntimeError('Nonfinite native SONIC turn reference')
    return channels, headings, yaw_rates
