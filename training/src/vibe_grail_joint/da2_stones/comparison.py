"""Matched ViBe/DA2 perception on the historical all-pillar training recipe."""
import copy


def enabled(settings):
    spec = settings.get('stones_comparison')
    if spec is None:
        return False
    if spec.get('version') != 1 or spec.get('method') not in ('vibe_matched', 'da2_ours', 'vibe_native_ppo'):
        raise ValueError('Unknown stones comparison protocol')
    if settings['initialization'] == 'raw_sonic_fresh_visual':
        if any(settings.get(k) for k in ('stage1_checkpoint', 'reference_checkpoint', 'repair_warm_start')):
            raise ValueError('Raw SONIC comparison excludes local stage-one checkpoints')
    elif settings['initialization'] != 'stage1_nonvisual_fresh_visual' or spec['method'] != 'vibe_matched':
        raise ValueError('Unsupported pillar comparison initialization')
    if spec.get('reference_mode') != 'all_flat_planner':
        raise ValueError('Pillar comparison requires terrain-blind flat planner references')
    if spec['method'] == 'vibe_native_ppo':
        from .native_vibe import validate_settings
        validate_settings(settings)
    elif spec['method'] == 'vibe_matched':
        if settings['visual']['backend'] != 'theia_native' or settings['geometry_aux']['coefficient'] != 0:
            raise ValueError('ViBe-M uses native Theia with no geometry auxiliary objective')
    elif settings['visual']['backend'] != 'da2_metric_small' or settings['geometry_aux']['coefficient'] != .1:
        raise ValueError('Full DA2 retains its original encoder and geometry objective')
    if any(settings.get(k) for k in ('flat_training', 'flat_dagger', 'flat_standing_dagger',
                                     'stones_stairs_style', 'layout_randomization', 'appearance_randomization',
                                     'lighting_randomization', 'groot_visual', 'dagger')):
        raise ValueError('Historical pillar comparison excludes later task/domain extensions')
    if settings['stones'].get('real_course') or settings['stones'].get('side_aware'):
        raise ValueError('This comparison uses the original ordinary cylindrical pillars')
    return True


def configure_env(cfg, settings):
    if not enabled(settings):
        return cfg
    if settings['stones_comparison']['method'] == 'vibe_native_ppo':
        from ..stairs_ab.native_vibe import configure_native_observations
        return configure_native_observations(cfg, settings)
    if settings['stones_comparison']['method'] == 'da2_ours':
        from ..stairs_ab.task import MatchedDA2Tokens
        cfg.observations['kv_tokens'].terms['img_tokens'].func = MatchedDA2Tokens
        return cfg
    from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
    from vibe.tasks.perloco.config.g1.observation_cfgs import ObsCtx, QUERY_GROUPS
    from ..stairs_ab.theia import TheiaTokens, cached_cls
    cfg.observations['kv_tokens'].terms = {'img_tokens': ObservationTermCfg(
        func=TheiaTokens, params={'visual': copy.deepcopy(settings['visual'])})}
    cfg.observations['q_task_cmd'] = QUERY_GROUPS['q_task_cmd'](ObsCtx(p={'command_name': 'motion'}))
    cfg.observations['q_cls'] = ObservationGroupCfg(
        terms={'img_cls': ObservationTermCfg(func=cached_cls)}, concatenate_terms=True, enable_corruption=False)
    return cfg


def configure_agent(cfg, bank, settings, n):
    if not enabled(settings):
        return cfg
    if settings['stones_comparison']['method'] == 'vibe_native_ppo':
        from .native_vibe import agent_config
        return agent_config(settings)
    if settings['initialization'] == 'raw_sonic_fresh_visual':
        cfg['critic']['class_name'] = 'vibe_grail_joint.stairs_ab.initialization.MatchedCritic'
        cfg['critic']['initialization_seed'] = settings['seed'] + 800000
    if settings['stones_comparison']['method'] == 'da2_ours':
        return cfg
    from ..velocity.config import agent_config
    native = agent_config(bank, settings, n)
    cfg['actor']['class_name'] = native['actor']['class_name']
    cfg['actor'].pop('geometry_aux', None)
    cfg['actor']['extractor_cfg'] = copy.deepcopy(native['actor']['extractor_cfg'])
    for extractor in cfg['actor']['extractor_cfg'].values():
        extractor['class_name'] = 'vibe_grail_joint.stairs_ab.theia.NativeTheiaFusion'
    cfg['obs_groups'] = copy.deepcopy(native['obs_groups'])
    return cfg


def audit_actor(actor, obs, settings):
    if not enabled(settings):
        return None
    if settings['stones_comparison']['method'] == 'vibe_native_ppo':
        from .native_vibe import audit_actor as audit_native_actor
        return audit_native_actor(actor, obs, settings)
    raw = settings['initialization'] == 'raw_sonic_fresh_visual'
    scope = ('Matched raw-SONIC ViBe-M/DA2 runs with the same pillar recipe' if raw else
             'Matched historical training recipe; old DA2 run is reused, not rerun')
    if settings['stones_comparison']['method'] == 'da2_ours':
        if not hasattr(actor, 'terrain_head') or tuple(obs['kv_tokens']['img_tokens'].shape[1:]) != (128, 96):
            raise RuntimeError('Full DA2 geometry head or two-frame token shape changed')
        return dict(method='DA2', task='original cylinder pillars', visual_backend='da2_metric_small',
                    patch_tokens=[128, 96], query_groups=['q_proprio', 'velocity_command'],
                    frames=2, heads=8, geometry_head=True, geometry_collection=True,
                    initialization='raw SONIC; fresh LoRA, visual fusion, command branch and geometry head',
                    comparison_scope=scope)
    if hasattr(actor, 'terrain_head') or hasattr(actor, 'auxiliary_loss'):
        raise RuntimeError('ViBe-M must not contain a DA2 geometry head')
    if tuple(obs['kv_tokens']['img_tokens'].shape[1:]) != (49, 192) or obs['q_cls'].shape[-1] != 192:
        raise RuntimeError('Native Theia patch/CLS shape mismatch')
    extractors = list(actor.extractors.values())
    expected = ('q_task_cmd', 'q_proprio', 'q_cls')
    if len(extractors) != 1 or tuple(extractors[0].input_groups[1:]) != expected:
        raise RuntimeError('ViBe-M native three-query fusion was changed')
    return dict(method='ViBe-M', task='original cylinder pillars', visual_backend='theia_native',
                patch_tokens=[49, 192], cls_width=192, query_groups=list(expected),
                frames=1, heads=1, geometry_head=False, geometry_collection=False,
                initialization=('raw SONIC; fresh LoRA, native Theia fusion and command branch' if raw else
                                'shared trained nonvisual stage1 tensors; fresh native Theia fusion and command branch'),
                comparison_scope=scope)
