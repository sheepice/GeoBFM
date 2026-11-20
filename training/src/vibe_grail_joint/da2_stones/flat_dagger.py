"""Stairs-style student-driven DAgger for standing, walking and starts/stops."""
import math

import torch

from ..velocity.dagger_buffer import DaggerBuffer
from ..velocity.dagger_env import DaggerVecEnv
from ..velocity.dagger_policy import build_teacher


def dagger_roles(cfg):
    if cfg and cfg['version'] == 2:
        return (1, 2, 3) if 'walk' in cfg['roles'] else (1, 3)
    return (1,) if cfg else ()


def role_mask(roles, selected):
    mask = torch.zeros_like(roles, dtype=torch.bool)
    for role in selected:
        mask |= roles == role
    return mask


def strata_count(cfg):
    if cfg['version'] == 2:
        return 5 if 'walk' in cfg['roles'] else 3
    return 1


def reference_strata(reference_ids, include_walk=False):
    """Balance the original SONIC reference types/speeds like stairs flat DAgger."""
    result = torch.full_like(reference_ids, -1)
    references = (0, 1, 2, 3, 4) if include_walk else (0, 3, 4)
    for key, reference in enumerate(references):
        result[reference_ids == reference] = key
    if (result < 0).any():
        raise ValueError('DAgger collected a reference outside the configured flat roles')
    return result


def validate_settings(settings):
    cfg = settings.get('flat_dagger')
    legacy = settings.get('flat_standing_dagger')
    if cfg is not None and legacy is not None:
        raise ValueError('Choose one flat DAgger recipe')
    if cfg is None:
        cfg = legacy
    if cfg is None:
        return None
    if not settings.get('flat_training'):
        raise ValueError('Flat DAgger requires the mixed flat task')
    if cfg.get('version') == 1:
        if settings.get('flat_dagger') is not None or cfg.get('reference') != 'sonic_planner_mode0_stand':
            raise ValueError('Version 1 is the original standing-only DAgger recipe')
    elif cfg.get('version') == 2:
        roles = cfg.get('roles')
        expected_reference = ('sonic_mode0_stand_walk_and_continuous_start_stop' if roles == ['stand', 'walk', 'start_stop']
                              else 'sonic_mode0_stand_and_continuous_start_stop')
        if (legacy is not None or roles not in (['stand', 'start_stop'], ['stand', 'walk', 'start_stop']) or
                cfg.get('reference') != expected_reference):
            raise ValueError('Version 2 requires the matching original SONIC flat references')
    else:
        raise ValueError('Unknown flat DAgger recipe version')
    coefficient = cfg.get('coefficient', -1)
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError('Invalid flat DAgger coefficient')
    for key in ('batch_size', 'buffer_capacity'):
        if not isinstance(cfg.get(key), int) or cfg[key] < 1:
            raise ValueError(f'Invalid flat DAgger {key}')
    if cfg['buffer_capacity'] < strata_count(cfg):
        raise ValueError('Flat DAgger buffer must cover every reference stratum')
    if cfg.get('teacher_checkpoint') != settings['sonic_checkpoint']:
        raise ValueError('Flat teacher must be the original frozen SONIC checkpoint')
    return cfg


def validate_actor_initialization(source, target):
    """Objective changes start fresh PPO state, retaining actor and terrain schedules."""
    cfg = validate_settings(target)
    previous = validate_settings(source)
    from .landing import normalize_landing_warm_start
    normalized, landing_changed = normalize_landing_warm_start(source, target) if 'stones' in target else (target, False)
    if cfg is None or (previous is not None and not (
            set(dagger_roles(previous)) < set(dagger_roles(cfg)) or (previous == cfg and landing_changed))):
        raise ValueError('Actor initialization requires a DAgger-role expansion or landing objective change')
    expected, original = dict(normalized), dict(source)
    for recipe in ('flat_dagger', 'flat_standing_dagger'):
        expected.pop(recipe, None)
        original.pop(recipe, None)
    expected['experiment_name'] = original['experiment_name']
    if expected != original:
        raise ValueError('Actor initialization changed settings beyond DAgger/landing objective/name')
    return landing_changed


class FlatDaggerEnv(DaggerVecEnv):
    """Reuse the exact stairs pre-step labels, student execution and PPO slicing.

    Wrap the full-world GeometryCollector so ray IDs remain simulator IDs.
    Configured flat worlds never enter PPO value/GAE/advantage statistics.
    """
    def __init__(self, wrapped, settings, rank):
        self.env = wrapped
        self.settings, self.rank = settings, rank
        self.device, self.num_actions = wrapped.device, wrapped.num_actions
        self.max_episode_length, self.clip_actions = wrapped.max_episode_length, wrapped.clip_actions
        self.dagger_cfg = validate_settings(settings)
        c = self.unwrapped.command_manager.get_term('motion')
        self.ppo_ids = c.ppo_mask.nonzero().flatten()
        self.dagger_ids = c.dagger_mask.nonzero().flatten()
        if not torch.equal(c.dagger_mask, role_mask(c.task_role, dagger_roles(self.dagger_cfg))):
            raise ValueError('Flat DAgger mask does not match the configured roles')
        self.num_envs = len(self.ppo_ids)
        self.buffer = DaggerBuffer(self.dagger_cfg['buffer_capacity'], strata_count(self.dagger_cfg), settings['seed'] + rank)
        self.actor = self.teacher = None
        self.teacher_sha256 = None
        self._full_obs = None

    def attach(self, algorithm, actor_cfg, obs_groups):
        self.actor = algorithm.get_policy()
        self.teacher, digest = build_teacher(self._full_obs[:1], actor_cfg, obs_groups,
                                            self.dagger_cfg['teacher_checkpoint'], self.device)
        self.teacher_sha256 = algorithm.standing_teacher_sha256 = digest
        algorithm.standing_dagger_buffer = self.buffer

    def state_dict(self):
        return dict(version=self.dagger_cfg['version'], teacher_sha256=self.teacher_sha256,
                    num_worlds=self.unwrapped.num_envs, replay=self.buffer.state_dict(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(self.device))

    def load_state_dict(self, state):
        if (state['version'], state['teacher_sha256'], state['num_worlds']) != (
                self.dagger_cfg['version'], self.teacher_sha256, self.unwrapped.num_envs):
            raise ValueError('Flat DAgger replay/teacher/world-count mismatch')
        self.buffer.load_state_dict(state['replay'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'], self.device)
