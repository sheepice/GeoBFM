"""Independent DA2 camera configuration over the established evaluation terrains."""
import copy
from dataclasses import fields
from ..common.scenes import cases_and_tiles


def evaluation_env(settings, *, seed, duration, heights, speeds, approaches, irregular=False,
                   record_case=None, limit_cases=None, route='up', diagnostic_geometry=False,
                   match_height_initialization=False, measure_foot_contact=False, extra_sensors=(), configure_scene=None):
    import mjlab
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from orcs.core.robustness import strip_domain
    from .task import env_config
    if settings.get('comparison'):
        from ..stairs_ab.task import env_config
    if settings.get('native_vibe') or settings.get('evaluation_baseline', {}).get('method') == 'vibe_original':
        from ..stairs_ab.native_vibe import env_config
    from ..velocity.eval_profile_command import ProfileEvaluationCommandCfg
    from ..velocity.step_quality_observer import StepQualityObserver
    from ..common.review_camera import add_review_camera
    settings = copy.deepcopy(settings); settings['seed'] = seed
    tiles, cases = cases_and_tiles(settings, seed, heights, speeds, approaches, irregular, route)
    if match_height_initialization:
        starts = {}
        for case in cases:
            key = (case['speed'], case['phase'], case['approach_extra_m'])
            starts.setdefault(key, case['spawn_xy'])
            case['spawn_xy'] = list(starts[key])
    if record_case is not None:
        if not 0 <= record_case < len(cases): raise ValueError('Recording case out of range')
        cases = [cases[record_case]]
    elif limit_cases is not None:
        if limit_cases < 1: raise ValueError('Case limit must be positive')
        cases = cases[:limit_cases]
    cfg = env_config(settings['bank_root'] + '/rank_0', settings, 20)
    if settings.get('evaluation_baseline', {}).get('method') in ('sonic_original', 'grail_mjlab_transfer'):
        for name in ('kv_tokens', 'q_task_cmd', 'q_cls'):
            cfg.observations.pop(name, None)
    if diagnostic_geometry:
        # Labels for offline inspection only: observation groups remain RGB-only.
        from dataclasses import replace
        cfg.scene.sensors = tuple(replace(s, data_types=('rgb', 'depth', 'segmentation'))
                                  if s.name == 'head_cam' else s for s in cfg.scene.sensors)
    cfg.scene.num_envs = len(cases)
    # This training-only weight is excluded from policy inputs and requires the
    # training sampler's masks, which fixed evaluation commands do not expose.
    cfg.observations.pop('flat_supervision', None)
    old = cfg.commands['motion']
    params = {f.name: getattr(old, f.name) for f in fields(ProfileEvaluationCommandCfg) if hasattr(old, f.name)}
    params.update(cases=tuple(cases), random_tiles=tuple(tiles), pose_range={}, velocity_range={}, joint_position_range=(0., 0.))
    cfg.commands['motion'] = ProfileEvaluationCommandCfg(**params)
    atlas = cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs']
    atlas.random_tiles = tuple(tiles); atlas.rows = ()
    if record_case is not None: add_review_camera(cfg, tiles[cases[0]['tile']])
    cfg.scene.sensors = (*cfg.scene.sensors, *extra_sensors)
    if configure_scene is not None:
        if record_case is None:
            raise ValueError("A custom recording scene requires one selected tile")
        configure_scene(cfg, tiles[cases[0]["tile"]])
    strip_domain(cfg)
    cfg.events.pop('policy_update_counter', None)
    for group in cfg.observations.values():
        group.enable_corruption = False
        for term in group.terms.values(): term.noise = None
    cfg.terminations.clear(); cfg.rewards = {}; cfg.episode_length_s = duration + 5
    if measure_foot_contact:
        from ..stairs_ab.evaluation_metrics import add_foot_pair_sensor
        add_foot_pair_sensor(cfg)

    class ObservedEnv(ManagerBasedRlEnv):
        def __init__(self, cfg, **kwargs):
            super().__init__(cfg, **kwargs)
            self.quality = StepQualityObserver(self)
            if measure_foot_contact:
                from ..stairs_ab.evaluation_metrics import FootPairObserver
                self.foot_pair = FootPairObserver(self)
            previous = self.metrics_manager.compute_substep
            def substep():
                previous(); self.quality.observe_substep()
                if measure_foot_contact: self.foot_pair.observe_substep()
            self.metrics_manager.compute_substep = substep

        def step(self, actions):
            self.quality.begin_step()
            if measure_foot_contact: self.foot_pair.begin_step()
            return super().step(actions)

        def _reset_idx(self, ids=None):
            if hasattr(self, 'quality'): self.quality.reset(ids)
            return super()._reset_idx(ids)

    env = RslRlVecEnvWrapper(ObservedEnv(cfg, device='cuda:0'))
    return env, settings, cases
