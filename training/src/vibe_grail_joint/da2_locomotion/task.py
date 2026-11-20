"""Assemble a new visual policy over the unchanged 50/20/15/15 task mixture."""
import copy

from mjlab.managers import ObservationTermCfg
from ..velocity.quality_task import env_config as original_env_config, QualityStairsEnv, REMOVED_REWARDS
from ..velocity.quality_task import options as original_options
from ..velocity.config import agent_config as original_agent_config
from .observation import FrozenVisualTokens


def task_fractions(settings):
    return dict(online_ppo=.5, paired_grail_ppo=.2, flat_dagger_planner=.15, flat_dagger_replay=.15)


def original_mixture(settings):
    return settings['task_mode'] == 'stairs_original_mixture_v1'


def options(settings):
    if not original_mixture(settings) or settings.get('gaussgym') or settings.get('hiking_perception_experiment'):
        raise ValueError('DA2 experiment requires the original mixture and no independent geometry branch')
    value = original_options(settings)
    value.update(visual=copy.deepcopy(settings['visual']), geometry_aux=copy.deepcopy(settings['geometry_aux']),
                 camera=copy.deepcopy(settings['camera']))
    if settings.get('render_domain'):
        from .render_domain import validate
        validate(settings['render_domain'])
        value['render_domain'] = copy.deepcopy(settings['render_domain'])
    if settings.get('groot_visual'):
        from .groot_domain import validate
        validate(settings['groot_visual'])
        value['groot_visual'] = copy.deepcopy(settings['groot_visual'])
    return value


def configure_visual_env(cfg, settings):
    visual = settings['visual']
    if settings['camera']['width'] != 112 or settings['camera']['height'] != 112:
        raise ValueError('This experiment uses a native 112x112 render without cropping')
    sensors = []
    for sensor in cfg.scene.sensors:
        if sensor.name == 'head_cam':
            from .camera import configure_camera
            sensor = configure_camera(sensor, settings['camera'])
        sensors.append(sensor)
    cfg.scene.sensors = tuple(sensors)
    cfg.observations['kv_tokens'].terms = {'img_tokens': ObservationTermCfg(
        func=FrozenVisualTokens, params={'visual': copy.deepcopy(visual)})}
    cfg.observations['kv_tokens'].enable_corruption = False
    cfg.observations.pop('q_cls', None)
    cfg.observations.pop('q_task_cmd', None)
    if settings.get('render_domain'):
        from .render_domain import configure
        cfg = configure(cfg, settings['render_domain'])
    if settings.get('groot_visual'):
        from .groot_domain import configure
        cfg = configure(cfg, settings)
    return cfg


def env_config(bank_path, settings, num_envs):
    options(settings)
    return configure_visual_env(original_env_config(bank_path, settings, num_envs), settings)


def agent_config(bank_path, settings, num_envs):
    agent = original_agent_config(bank_path, settings, num_envs)
    visual = settings['visual']
    agent['experiment_name'] = 'da2_locomotion'
    agent['actor']['class_name'] = 'vibe_grail_joint.da2_locomotion.model.TerrainConditionedActor'
    agent['actor']['geometry_aux'] = settings['geometry_aux']
    # Source-only distributions keep frozen SONIC weights outside the vendor tree.
    agent['actor']['base_checkpoint'] = (
        (settings.get('dagger') or {}).get('teacher_checkpoint') or settings['sonic_checkpoint']
    )
    for group in agent['actor']['extractor_cfg']:
        agent['actor']['extractor_cfg'][group] = dict(
            class_name='vibe_grail_joint.da2_locomotion.model.VisualQueryExtractor',
            query_groups=['q_proprio', 'velocity_command'], frames=visual['frames'],
            patch_grid=visual['patch_grid'], num_heads=visual['num_heads'],
            attn_dim=visual['attention_width'], latent_dim=128)
    for name, groups in agent['obs_groups'].items():
        agent['obs_groups'][name] = [g for g in groups if g not in ('q_cls', 'q_task_cmd')]
    agent['algorithm']['class_name'] = 'vibe_grail_joint.da2_locomotion.algorithm.GeometryPPO'
    agent['algorithm']['geometry_aux'] = settings['geometry_aux']
    return agent
