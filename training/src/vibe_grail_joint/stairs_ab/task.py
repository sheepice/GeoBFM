"""One physical task and training recipe, two native visual architectures."""
import copy
from dataclasses import dataclass, fields

import torch
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg

from ..da2_flat_repair import task as common
from ..velocity.config import agent_config as native_agent_config
from ..velocity.dagger_task import DaggerMotionCommand, DaggerMotionCommandCfg
from ..velocity.recovery_terrain import ProfileMotionCommand
from ..da2_locomotion.observation import FrozenVisualTokens

QualityStairsEnv = common.QualityStairsEnv
REMOVED_REWARDS = common.REMOVED_REWARDS
original_mixture = common.original_mixture


class MatchedDA2Tokens(FrozenVisualTokens):
    def __init__(self, cfg, env):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            super().__init__(cfg, env)


def flat_reference_masks(num_envs, device='cpu', stairs_only=False):
    if num_envs < 20 or num_envs % 20:
        raise ValueError('Matched task requires complete blocks of 20 worlds')
    slot = torch.arange(num_envs, device=device) % 20
    if stairs_only:
        yes = torch.ones_like(slot, dtype=torch.bool)
        no = torch.zeros_like(yes)
        return yes, no.clone(), no.clone(), no.clone()
    return slot < 14, torch.zeros_like(slot, dtype=torch.bool), (slot >= 14) & (slot < 17), slot >= 17


class FlatReferenceCommand(DaggerMotionCommand):
    def _init_task(self):
        if self.cfg.stairs_only:
            ProfileMotionCommand._init_task(self)
        else:
            super()._init_task()
        self.ppo_online, self.paired, self.dagger_planner, self.flat = flat_reference_masks(
            self.num_envs, self.device, self.cfg.stairs_only)
        self.dagger_mask = self.dagger_planner | self.flat
        self.ppo_mask = self.ppo_online
        self.online = self.ppo_online | self.dagger_planner
        self.online_ids = self.online.nonzero().flatten()
        self.metrics['terrain_fraction'] = self.ppo_online.float()

    def _sample_init_frame(self, env_ids):
        chosen, frames = super()._sample_init_frame(env_ids)
        if self.cfg.stairs_only:
            # Preserve the old stair-direction ratios, conditioned on stairs.
            # Empty flat/paired masks ensure no replay clip is ever selected.
            flat = self.tile_data['kind'][self.tile[env_ids]] == 3
            ids = env_ids[flat]
            if len(ids):
                old_tile = self.tile[ids].clone()
                reference_xy = self.motion.body_pos_w[frames[flat], 0, :2]
                local_xy = self._env.scene.env_origins[ids, :2] + reference_xy - self.tile_origins[old_tile, :2]
                kinds = torch.multinomial(torch.tensor([.35, .25, .25], device=self.device), len(ids), replacement=True)
                self.tile[ids] = (old_tile // 4) * 4 + kinds
                spawn_xy = self.tile_origins[self.tile[ids], :2] + local_xy
                self._env.scene.env_origins[ids, :2] = spawn_xy - reference_xy
                self._env.scene.env_origins[ids, 2] = self.ground_height(spawn_xy, ids)
        return chosen, frames


@dataclass(kw_only=True)
class FlatReferenceCommandCfg(DaggerMotionCommandCfg):
    stairs_only: bool = False

    def build(self, env):
        return FlatReferenceCommand(self, env)


def task_fractions(settings):
    if settings['comparison'].get('stairs_only'):
        return dict(online_ppo=1., paired_grail_ppo=0., flat_dagger_planner=0., flat_dagger_replay=0.)
    return dict(online_ppo=.7, paired_grail_ppo=0., flat_dagger_planner=.15, flat_dagger_replay=.15)


def options(settings):
    from da2_vibe.ablations import validate_ablation
    validate_ablation(settings)
    if settings['initialization'] != 'raw_sonic_fresh_visual':
        raise ValueError('A/B must start from original SONIC with fresh adaptation')
    if any(settings.get(key) for key in ('stage1_checkpoint', 'repair_warm_start', 'render_domain', 'groot_visual', 'stair_failures')):
        raise ValueError('A/B excludes local stage-one weights and later real-robot adaptations')
    if settings['comparison']['reference_mode'] != 'all_flat_planner':
        raise ValueError('All PPO references must be terrain-blind flat planner references')
    if settings['comparison'].get('stairs_only') and settings.get('dagger') is not None:
        raise ValueError('Stairs-only A/B must not contain flat DAgger data')
    value = common.options(settings)
    value['comparison'] = copy.deepcopy(settings['comparison'])
    if settings.get('ablation'):
        value['ablation'] = copy.deepcopy(settings['ablation'])
    return value


def env_config(bank_path, settings, num_envs):
    options(settings)
    cfg = common.env_config(bank_path, settings, num_envs)
    old = cfg.commands['motion']
    cfg.commands['motion'] = FlatReferenceCommandCfg(**{f.name: getattr(old, f.name) for f in fields(DaggerMotionCommandCfg)},
                                                    stairs_only=settings['comparison'].get('stairs_only', False))
    cfg.events.pop('rand_light_dir', None)
    if settings['comparison']['method'] == 'vibe_matched':
        from vibe.tasks.perloco.config.g1.observation_cfgs import ObsCtx, QUERY_GROUPS
        from .theia import TheiaTokens, cached_cls
        cfg.observations['kv_tokens'].terms = {'img_tokens': ObservationTermCfg(func=TheiaTokens, params={'visual': settings['visual']})}
        cfg.observations['q_task_cmd'] = QUERY_GROUPS['q_task_cmd'](ObsCtx(p={'command_name': 'motion'}))
        cfg.observations['q_cls'] = ObservationGroupCfg(terms={'img_cls': ObservationTermCfg(func=cached_cls)},
                                                       concatenate_terms=True, enable_corruption=False)
    elif settings['comparison']['method'] != 'da2_ours':
        raise ValueError('Unknown A/B method')
    else:
        cfg.observations['kv_tokens'].terms['img_tokens'].func = MatchedDA2Tokens
        if settings.get('ablation') and settings['visual']['ablation_input'] != 'normal':
            from .ablation import AblationDA2Tokens
            cfg.observations['kv_tokens'].terms['img_tokens'].func = AblationDA2Tokens
    return cfg


def agent_config(bank_path, settings, num_envs):
    cfg = common.agent_config(bank_path, settings, num_envs)
    if settings['comparison']['method'] == 'vibe_matched':
        native = native_agent_config(bank_path, settings, num_envs)
        cfg['actor']['class_name'] = native['actor']['class_name']
        cfg['actor'].pop('geometry_aux', None)
        cfg['actor']['extractor_cfg'] = native['actor']['extractor_cfg']
        for extractor in cfg['actor']['extractor_cfg'].values():
            extractor['class_name'] = 'vibe_grail_joint.stairs_ab.theia.NativeTheiaFusion'
        cfg['obs_groups'] = native['obs_groups']
    cfg['experiment_name'] = 'stairs_ab_' + settings['comparison']['method']
    if settings.get('ablation'):
        for extractor in cfg['actor']['extractor_cfg'].values():
            extractor['query_mode'] = settings['visual']['query_mode']
        cfg['experiment_name'] = 'stairs_ablation_' + settings['ablation']['variant']
    cfg['critic']['class_name'] = 'vibe_grail_joint.stairs_ab.initialization.MatchedCritic'
    cfg['critic']['initialization_seed'] = settings['seed'] + 800000
    if settings['comparison'].get('stairs_only'):
        cfg['algorithm'].update(dagger_coef=0., dagger_batch_size=256)
    return cfg
