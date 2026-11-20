"""Additive corrected second-stage task. Original task/configs remain reproducible."""
import copy
import math
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from .normal_stairs_task import (env_config as base_env_config, options as base_options,
                                 agent_config, guidance_reward, REMOVED_REWARDS)
from .quality_runtime import QualityGuidance
from .step_quality_observer import add_quality_sensors


def options(settings):
    value=base_options(settings)
    quality=settings['step_quality']
    if quality['version']!=1:raise ValueError('Unknown step-quality revision')
    for key,number in quality.items():
        if not isinstance(number,(int,float)) or not math.isfinite(number):raise ValueError(f'Invalid {key}')
        if key.endswith('_weight'):
            if number>=0:raise ValueError(f'Quality costs need negative weights: {key}')
        elif number<=0:raise ValueError(f'Positive quality option required: {key}')
    value['step_quality']=copy.deepcopy(quality)
    if settings.get('stair_failures'):
        from .stair_failure import validate
        validate(settings['stair_failures'])
        value['stair_failures'] = copy.deepcopy(settings['stair_failures'])
    return value


def env_config(bank_path,settings,num_envs):
    options(settings)
    cfg=base_env_config(bank_path,settings,num_envs)
    add_quality_sensors(cfg)
    # Replace the ankle-height proxy. Keep the name for downstream task contracts.
    cfg.rewards['online_slip']=RewardTermCfg(func=guidance_reward,
        weight=settings['step_quality']['contact_slip_weight'],params={'name':'contact_slip'})
    for name in ('riser_impact','shin_impact','swing_path','support_margin','hard_landing'):
        cfg.rewards[f'online_{name}']=RewardTermCfg(func=guidance_reward,
            weight=settings['step_quality'][f'{name}_weight'],params={'name':name})
    if settings.get('stair_failures'):
        from .stair_failure import add_sensors, edge_termination, pair_termination, failure_cost, guarded_success
        add_sensors(cfg)
        cfg.terminations['stair_edge_support'] = TerminationTermCfg(func=edge_termination)
        cfg.terminations['left_right_foot_collision'] = TerminationTermCfg(func=pair_termination)
        success = cfg.terminations['online_success']
        fall = cfg.terminations['online_fall']
        success.params = {'base_func': success.func, 'base_params': copy.deepcopy(success.params),
                          'fall_func': fall.func, 'fall_params': copy.deepcopy(fall.params)}
        success.func = guarded_success
        reward = cfg.rewards['online_fall']
        reward.params = {'base_func': reward.func, 'base_params': copy.deepcopy(reward.params)}
        reward.func = failure_cost
    return cfg


class QualityStairsEnv(ManagerBasedRlEnv):
    def __init__(self,cfg,*,device,settings):
        super().__init__(cfg,device=device)
        self.guidance=QualityGuidance(self,settings)
        if settings.get('stair_failures'):
            from .stair_failure import StairFailureObserver
            self.stair_failures = StairFailureObserver(self, self.guidance.quality, settings['stair_failures'])
        previous=self.metrics_manager.compute_substep
        def substep():
            previous()
            self.guidance.quality.observe_substep()
            if hasattr(self, 'stair_failures'):
                self.stair_failures.observe_substep()
        self.metrics_manager.compute_substep=substep

    def step(self,actions):
        self.guidance.before_step(actions)
        if hasattr(self, 'stair_failures'):
            self.stair_failures.begin_step()
        result = super().step(actions)
        if hasattr(self, 'stair_failures'):
            obs, reward, terminated, truncated, extras = result
            # RSL-RL bootstraps timeouts, so a simultaneous true failure wins.
            return obs, reward, terminated, truncated & ~terminated, extras
        return result

    def _reset_idx(self,env_ids=None):
        if hasattr(self,'guidance'):self.guidance.reset(env_ids)
        if hasattr(self, 'stair_failures'):self.stair_failures.reset(env_ids)
        return super()._reset_idx(env_ids)
