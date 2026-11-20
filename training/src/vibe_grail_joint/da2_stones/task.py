"""100% online round-pillar PPO with the DA2 deployment interface unchanged."""
import copy
import math
from dataclasses import fields
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from ..velocity.config import env_config as velocity_env_config
from ..velocity.step_quality_observer import add_quality_sensors
from ..velocity.posture_task import posture_online_fall
from ..velocity.ssr_task import REMOVED_REWARDS
from ..da2_locomotion.task import configure_visual_env, agent_config as da2_agent_config
from .command import StoneCommandCfg
from .terrain import make_tiles, atlas_class, audit_tiles
from .runtime import StoneGuidance


def guidance_reward(env, name):
    value = env.guidance.reward(name)
    c = env.command_manager.get_term('motion')
    if hasattr(c, 'pillar_task') and name in ('alternate', 'wrong_step', 'low_support', 'bridge', 'same_top', 'reach', 'crossover', 'wrong_side_swing', 'swing_clearance'):
        value = value * c.pillar_task * (c.speed > .001)
    return value


def failed(env):
    c = env.command_manager.get_term('motion')
    fallen = posture_online_fall(env, half_width=2., margin=.5, minimum_clearance=.32, gravity_z=-.45)
    if hasattr(env, 'guidance'):
        env.guidance.measure()
        fallen |= env.guidance.floor_failure
        fallen |= env.guidance.side_failure
    c.terminal_fall = fallen.clone()
    return fallen


def failure_cost(env):
    return failed(env).float()


def succeeded(env):
    c = env.command_manager.get_term('motion')
    if not hasattr(env, 'guidance'):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.guidance.measure()
    root = c.robot.data.root_link_pos_w
    feet = c.robot_body_pos_w[:, [3, 6]]
    goal = c.tile_data['goal'][c.tile]
    origin = c.tile_origins[c.tile]
    passed = env.guidance.sequence.progress >= c.tile_data['n'][c.tile]
    passed &= root[:, 0] - origin[:, 0] >= goal
    passed &= (feet[:, :, 0] - origin[:, None, 0]).amin(-1) >= goal - .45
    passed &= c.robot.data.projected_gravity_b[:, 2] < -.85
    passed &= env.guidance.quality.sensor_support.any(-1)
    if hasattr(c, 'route_targets'):
        new_level = c.tile_data['level'][c.tile] == 10
        completed = env.guidance.sequence.progress >= c.route_count[c.tile]
        feet_local = feet[:, :, :2] - origin[:, None, :2]
        on_exit = (feet_local[:, :, 0] >= c.route_exit[c.tile, None]).all(-1)
        on_exit &= feet_local[:, :, 1].abs().amax(-1) <= .55
        passed = torch.where(new_level,
            completed & on_exit & env.guidance.exit_ground & (c.speed > .001)
            & (root[:, 0] - origin[:, 0] >= goal)
            & (c.robot.data.projected_gravity_b[:, 2] < -.85), passed)
    passed &= ~env.guidance.floor_failure
    passed &= ~env.guidance.side_failure
    if c.success_step != int(env.common_step_counter):
        c.success_hold[:] = torch.where(passed, c.success_hold + 1, 0)
        c.success_step = int(env.common_step_counter)
    c.terminal_success = c.success_hold >= 10
    if hasattr(c, 'pillar_task'):
        c.terminal_success &= c.pillar_task
    return c.terminal_success


def stalled(env):
    if not hasattr(env, 'guidance'):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.guidance.measure()
    c = env.command_manager.get_term('motion')
    active = c.pillar_task & (c.speed > .001) if hasattr(c, 'pillar_task') else c.online
    return (env.guidance.steps_since_progress > 500) & active


@torch.no_grad()
def flat_supervision(env):
    c = env.command_manager.get_term('motion')
    feet = c.robot_body_pos_w[:, [3, 6]]
    x = feet[:, :, 0] - c.tile_origins[c.tile, None, 0]
    before = x.amax(-1) < c.course_start[c.tile] - .6
    after = x.amin(-1) > c.course_end[c.tile] + .6
    weight = before | after
    if hasattr(c, 'flat_task'):
        # Independent flat worlds get their own all-joint teacher loss.
        weight &= c.pillar_task & (c.speed > .001)
    return weight.float()[:, None]


def flat_teacher_supervision(env):
    c = env.command_manager.get_term('motion')
    return (c.flat_task & ~c.dagger_mask).float()[:, None]


def flat_timeout(env):
    c = env.command_manager.get_term('motion')
    return c.flat_task & (c.reference_step >= round(c.cfg.flat_training['episode_seconds'] / env.step_dt))


def idle_stability(env):
    c = env.command_manager.get_term('motion')
    idle = (c.speed < .001) & (c.idle_steps >= 40)
    v = c.robot.data.root_link_lin_vel_w[:, :2].square().sum(-1)
    w = c.robot.data.root_link_ang_vel_w[:, :2].square().sum(-1)
    return torch.exp(-v / .1**2 - w / .3**2) * idle


def idle_drift(env):
    c = env.command_manager.get_term('motion')
    error = (c.robot.data.root_link_pos_w[:, :2] - c.idle_anchor).square().sum(-1)
    return (error / .15**2).clamp_max(4) * (c.speed < .001) * (c.idle_steps >= 40)


def idle_joint_velocity(env):
    c = env.command_manager.get_term('motion')
    return c.robot_joint_vel.square().mean(-1) * (c.speed < .001) * (c.idle_steps >= 40)


def env_config(bank, settings, n, rank=0):
    from .landing import validate_settings as validate_landing
    validate_landing(settings)
    from .stairs_style import enabled, validate_settings
    validate_settings(settings)
    stairs_style = enabled(settings)
    adapted = copy.deepcopy(settings)
    adapted['dagger'] = None
    cfg = velocity_env_config(bank, adapted, n)
    level10 = settings.get('level10')
    base_levels = settings['stones']['levels']
    if level10:
        if base_levels != 11 or not (stairs_style and settings.get('layout_randomization')
                                    and settings['stones'].get('real_course')):
            raise ValueError('Level 10 extends the ten existing yellow courses in the stairs-style recipe')
        base_levels -= 1
    tiles = tuple(make_tiles(settings['seed'] + 7919 * rank,
                             base_levels, settings['profile_variants'],
                             settings['stones'].get('real_course'),
                             settings['stones'].get('side_aware', {}).get('minimum_stagger_m', 0.)))
    mixed = settings.get('flat_training')
    if mixed or stairs_style:
        from .terrain import append_flat_tiles
        tiles = append_flat_tiles(tiles)
    audit_tiles(tiles)
    physical_tiles=tiles
    layout=settings.get('layout_randomization')
    if layout:
        from .layout_randomization import build_layout_bank
        if not settings['stones'].get('local_coordinates'):
            raise ValueError('Layout randomization requires local coordinates')
        tiles=build_layout_bank(physical_tiles,base_levels,settings['profile_variants'],
            settings['seed']+7919*rank,layout)
    if level10:
        from .level10 import append_level10
        physical_tiles, tiles = append_level10(physical_tiles, tiles, settings['seed'] + 7919 * rank, level10)
    old = cfg.commands['motion']
    command_cfg = StoneCommandCfg
    if stairs_style:
        from .stairs_style import StairsStyleStoneCommandCfg
        command_cfg = StairsStyleStoneCommandCfg
    if mixed:
        from .mixed_command import MixedStoneCommandCfg
        command_cfg = MixedStoneCommandCfg
    params = {f.name: getattr(old, f.name) for f in fields(command_cfg) if hasattr(old, f.name)}
    params.update(random_tiles=tiles, stone_levels=settings['stones']['levels'],
                  fixed_speed=settings['stones']['speed_m_s'], seed=settings['seed'] + rank,
                  local_coordinates=settings['stones'].get('local_coordinates', False))
    if layout:
        params.update(physical_tiles=physical_tiles,
            profile_variants=settings['profile_variants']*layout['layouts_per_variant'],
            approach_spawn_x=tuple(layout['approach_spawn_x_m']),layout_settings=copy.deepcopy(layout))
    if level10:
        params['level10'] = copy.deepcopy(level10)
    if settings.get('flat_turning'):
        params['flat_turning'] = copy.deepcopy(settings['flat_turning'])
    if mixed:
        params['flat_training'] = copy.deepcopy(mixed)
        from .flat_dagger import validate_settings, dagger_roles
        dagger = validate_settings(settings)
        params['standing_dagger'] = dagger is not None
        params['flat_dagger_roles'] = dagger_roles(dagger)
    cfg.commands['motion'] = command_cfg(**params)
    # No paired heightfields, stair solids or hidden raised connecting platforms.
    cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs'] = atlas_class()(rows=(), random_tiles=physical_tiles)
    cfg = configure_visual_env(cfg, settings)
    if settings['stones'].get('real_course'):
        # Preserve yellow mats/pillars and gray floor instead of recoloring them.
        cfg.events.pop('rand_terrain_color', None)
        cfg.events.pop('set_terrain_color', None)
    if settings.get('appearance_randomization'):
        from mjlab.managers import EventTermCfg
        from .appearance import randomize_appearance, validate_settings
        validate_settings(settings['appearance_randomization'])
        cfg.events.pop('rand_terrain_color', None)
        cfg.events.pop('set_terrain_color', None)
        cfg.events['stone_appearance'] = EventTermCfg(func=randomize_appearance,
            mode='reset', params={'settings': copy.deepcopy(settings['appearance_randomization'])})
    if settings.get('lighting_randomization'):
        from mjlab.managers import EventTermCfg
        from .appearance import randomize_lighting, validate_lighting_settings
        validate_lighting_settings(settings['lighting_randomization'])
        cfg.events.pop('rand_light_dir',None)
        cfg.events['stone_lighting']=EventTermCfg(func=randomize_lighting,mode='reset',
            params={'settings':copy.deepcopy(settings['lighting_randomization'])})
    add_quality_sensors(cfg)
    for name in list(cfg.rewards):
        if name in REMOVED_REWARDS or name.startswith('motion_') or cfg.rewards[name].func.__name__ == 'replay_reward':
            cfg.rewards.pop(name)
    cfg.rewards.pop('online_slip', None)
    cfg.rewards['online_fall'] = RewardTermCfg(func=failure_cost, weight=-12.)
    for name, weight in settings['stones']['reward_weights'].items():
        cfg.rewards['stone_' + name] = RewardTermCfg(func=guidance_reward, weight=weight, params={'name': name})
    cfg.terminations = {'time_out': cfg.terminations['time_out'],
        'stone_failure': TerminationTermCfg(func=failed),
        'stone_success': TerminationTermCfg(func=succeeded, time_out=True),
        'stone_stall': TerminationTermCfg(func=stalled)}
    cfg.observations['flat_supervision'] = ObservationGroupCfg(
        terms={'weight': ObservationTermCfg(func=flat_supervision)}, concatenate_terms=True, enable_corruption=False)
    if stairs_style:
        from .stairs_style import flat_supervision as stairs_flat_supervision
        cfg.observations['flat_supervision'].terms['weight'] = ObservationTermCfg(
            func=stairs_flat_supervision, params={'settings': settings['flat_repair']})
        cfg.rewards['idle_stability'] = RewardTermCfg(func=idle_stability, weight=1.)
        cfg.rewards['idle_drift'] = RewardTermCfg(func=idle_drift, weight=-.5)
        cfg.rewards['idle_joint_velocity'] = RewardTermCfg(func=idle_joint_velocity, weight=-.02)
    if mixed:
        cfg.observations['flat_teacher_supervision'] = ObservationGroupCfg(
            terms={'weight': ObservationTermCfg(func=flat_teacher_supervision)}, concatenate_terms=True, enable_corruption=False)
        cfg.terminations['flat_timeout'] = TerminationTermCfg(func=flat_timeout, time_out=True)
        cfg.rewards['idle_stability'] = RewardTermCfg(func=idle_stability, weight=1.)
        cfg.rewards['idle_drift'] = RewardTermCfg(func=idle_drift, weight=-.5)
        cfg.rewards['idle_joint_velocity'] = RewardTermCfg(func=idle_joint_velocity, weight=-.02)
    cfg.episode_length_s = 35.
    cfg.seed = settings['seed'] + rank
    if settings.get('stones_comparison'):
        from .comparison import configure_env
        cfg = configure_env(cfg, settings)
    if settings.get('flat_turning'):
        from .turning import flat_turn_timeout
        cfg.terminations['flat_turn_timeout'] = TerminationTermCfg(func=flat_turn_timeout, time_out=True)
    return cfg


def agent_config(bank, settings, n):
    adapted = copy.deepcopy(settings)
    from .stairs_style import enabled, validate_settings
    validate_settings(settings)
    stairs_style = enabled(settings)
    if not stairs_style:
        adapted['dagger'] = None
    agent = da2_agent_config(bank, adapted, n)
    agent['experiment_name'] = settings.get('experiment_name', 'da2_stones_stage1_20261101')
    agent['actor']['base_checkpoint'] = settings['sonic_checkpoint']
    agent['algorithm']['class_name'] = 'vibe_grail_joint.da2_stones.algorithm.StonePPO'
    agent['algorithm']['flat_repair'] = settings['flat_repair']
    if settings.get('flat_training'):
        agent['algorithm']['flat_training'] = settings['flat_training']
    from .flat_dagger import validate_settings, dagger_roles
    standing = validate_settings(settings)
    if standing:
        agent['algorithm']['flat_dagger' if standing['version'] == 2 else 'standing_dagger'] = standing
    dagger_slots = sum({1: 3, 2: 2, 3: 3}[role] for role in dagger_roles(standing))
    ppo_n = n - (n // 20) * dagger_slots
    agent['algorithm']['num_mini_batches'] = max(1, math.ceil(ppo_n * agent['num_steps_per_env'] / settings['minibatch_env_steps']))
    if stairs_style:
        agent['algorithm']['class_name'] = 'vibe_grail_joint.da2_flat_repair.algorithm.FlatRepairPPO'
        ppo_n = 3 * n // 5
        agent['algorithm']['num_mini_batches'] = max(1, math.ceil(ppo_n * agent['num_steps_per_env'] / settings['minibatch_env_steps']))
    assert all('flat_supervision' not in v for v in agent['obs_groups'].values())
    assert all('flat_teacher_supervision' not in v for v in agent['obs_groups'].values())
    if settings.get('stones_comparison'):
        from .comparison import configure_agent
        agent = configure_agent(agent, bank, settings, n)
    return agent


class StoneEnv(ManagerBasedRlEnv):
    def __init__(self, cfg, *, device, settings):
        self.level10_colors = settings.get('groot_visual', {}).get('level10_appearance', False)
        super().__init__(cfg, device=device)
        self.settings_stairs_style = settings.get('stones_stairs_style') is not None
        self.guidance = StoneGuidance(self, settings)
        self.completed_episodes = self.completed_successes = self.completed_falls = 0
        self.completed_progress_sum = 0.
        self.flat_episodes = self.flat_falls = 0
        self.obstacle_idle_episodes = self.obstacle_idle_falls = 0
        self.flat_stand_completed = self.flat_start_stop_completed = 0
        self.route_results = torch.zeros((3, 3), device=self.device, dtype=torch.long)
        previous = self.metrics_manager.compute_substep
        def substep():
            previous()
            self.guidance.quality.observe_substep()
        self.metrics_manager.compute_substep = substep

    def step(self, actions):
        if not torch.isfinite(actions).all():
            raise RuntimeError('Nonfinite policy action')
        self._diagnostic_last_actions = actions.detach()
        self.guidance.before_step()
        result = super().step(actions)
        c = self.command_manager.get_term('motion')
        if not torch.isfinite(c.robot.data.root_link_pos_w).all() or not torch.isfinite(c.robot_joint_pos).all():
            raise RuntimeError('Nonfinite robot physics; not masked by resetting')
        return result

    def _reset_idx(self, env_ids=None):
        if hasattr(self, 'guidance'):
            ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
            c = self.command_manager.get_term('motion')
            ended = ids[self.episode_length_buf[ids] > 0]
            if hasattr(c, 'pillar_task'):
                flat = ended[c.flat_task[ended]]
                if len(flat):
                    self.flat_episodes += len(flat)
                    self.flat_falls += int(c.terminal_fall[flat].sum())
                    if hasattr(c, 'reference_step'):
                        complete = (c.reference_step[flat] >= round(c.cfg.flat_training['episode_seconds'] / self.step_dt)) & ~c.terminal_fall[flat]
                        self.flat_stand_completed += int((complete & (c.task_role[flat] == 1)).sum())
                        self.flat_start_stop_completed += int((complete & (c.task_role[flat] == 3)).sum())
                ended = ended[c.pillar_task[ended]]
                if self.settings_stairs_style:
                    idle = ended[c.speed[ended] < .001]
                    self.obstacle_idle_episodes += len(idle)
                    self.obstacle_idle_falls += int(c.terminal_fall[idle].sum())
                    ended = ended[c.speed[ended] > .001]
            if len(ended):
                level = c.tile_data['level'][c.tile[ended]]
                if hasattr(c, 'route_kind'):
                    ids10 = ended[level == 10]
                    result = torch.stack((torch.ones_like(ids10), c.terminal_success[ids10].long(),
                                          c.terminal_fall[ids10].long()), -1)
                    self.route_results.index_add_(0, c.route_kind[c.tile[ids10]], result)
                values = torch.stack((torch.ones(len(ended), device=self.device),
                    c.terminal_success[ended].float(), self.guidance.terminal_progress[ended]), -1)
                c.curriculum_counts.index_add_(0, level, values)
                self.completed_episodes += len(ended)
                self.completed_successes += int(c.terminal_success[ended].sum())
                self.completed_falls += int(c.terminal_fall[ended].sum())
                self.completed_progress_sum += float(self.guidance.terminal_progress[ended].sum())
            self.guidance.reset(ids)
        result = super()._reset_idx(env_ids)
        if self.level10_colors:
            from ..da2_locomotion.groot_domain import apply_level10_appearance
            apply_level10_appearance(self, env_ids)
        return result
