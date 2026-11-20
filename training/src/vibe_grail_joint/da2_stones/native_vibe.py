"""Author ViBe actor and plain PPO, initialized from SONIC on round pillars."""


def validate_settings(settings):
    if (settings['initialization'] != 'raw_sonic_fresh_visual'
            or settings['task_mode'] != 'cylinder_stones_only_v1'
            or settings['visual']['backend'] != 'theia_native'
            or settings['geometry_aux']['coefficient'] != 0
            or settings['flat_repair']['coefficient'] != 0
            or settings['posture_revision']['upper_preservation_coef'] != 0
            or settings['preservation_coef'] != 0):
        raise ValueError('Native pillar PPO requires raw SONIC, Theia and zero auxiliary/preservation losses')
    if any(settings.get(k) for k in ('stage1_checkpoint', 'reference_checkpoint',
            'repair_warm_start', 'dagger', 'flat_training', 'flat_dagger', 'flat_standing_dagger')):
        raise ValueError('Native pillar PPO excludes stage-one and teacher-training objectives')
    if settings['learning_rate'] != .001 or settings['learning_rate_schedule'] != 'adaptive':
        raise ValueError('Native PPO uses the author adaptive learning-rate recipe')


def agent_config(settings):
    from ..stairs_ab.native_vibe import build_native_agent
    validate_settings(settings)
    return build_native_agent(settings, settings['experiment_name'])


def audit_actor(actor, obs, settings):
    from ..stairs_ab.native_vibe import NativeSonicActor
    validate_settings(settings)
    if not isinstance(actor, NativeSonicActor):
        raise RuntimeError('Expected native ViBe actor')
    if hasattr(actor, 'command_encoder') or hasattr(actor, 'terrain_head'):
        raise RuntimeError('Native ViBe must not gain the matched command branch or geometry head')
    if obs['augmentation'].shape[-1] != 6 or tuple(obs['kv_tokens']['img_tokens'].shape[1:]) != (49, 192):
        raise RuntimeError('Native ViBe observation contract changed')
    query_groups = ('q_task_cmd', 'q_proprio', 'q_cls')
    extractors = list(actor.extractors.values())
    if len(extractors) != 1 or tuple(extractors[0].input_groups[1:]) != query_groups:
        raise RuntimeError('Native ViBe query groups changed')
    return dict(method='ViBe native PPO', task='original cylinder pillars',
        visual_backend='theia_native', patch_tokens=[49, 192], query_groups=list(query_groups),
        frames=1, heads=1, geometry_head=False, geometry_collection=False,
        added_command_encoder=False, native_augmentation_width=6,
        initialization='raw SONIC; fresh LoRA and native Theia fusion',
        optimizer_recipe='author plain PPO; no auxiliary or preservation objectives')
