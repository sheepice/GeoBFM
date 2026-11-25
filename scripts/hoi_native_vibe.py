"""ViBe UoLM policy on the same HOI observations and compensated PD interface as GeoBFM."""

from dataclasses import asdict

import torch
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from vibe.tasks.uolm.config.g1.agent_cfgs import adapt_sonic_ext_agent_cfg
from vibe_grail_joint.stairs_ab.native_vibe import NativePpoFusion
from vibe_grail_joint.stairs_ab.theia import TheiaTokens, cached_cls
from rsl_rl.models import ExtractorSonicAdapterModel
from hoi_carry_task import motion, orientation6
from hoi_pickup_vision import PickupVisualTokens


class CarryTheiaTokens(TheiaTokens):
    """Only Theia features enter the policy. Share the original pickup controller in phase two."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.locator = PickupVisualTokens(cfg, env) if env.cfg.commands["motion"].visual_pickup else None
        env.da2_visual = self

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if self.locator is not None:
            self.locator.reset(env_ids)

    def __call__(self, env, **kwargs):
        if self.locator is not None:
            self.locator(env, **kwargs)
        return super().__call__(env, **kwargs)


def goal_pose(env):
    c = motion(env)
    q = c.box_pose[-1, 3:].expand(c.num_envs, -1)
    return torch.cat((orientation6(q), c.box_pose[-1, :3].expand(c.num_envs, -1)), dim=-1)


def native_observations(cfg):
    visual = cfg.observations.pop("kv_tokens")
    visual.terms["img_tokens"].func = CarryTheiaTokens

    def group(fun):
        return ObservationGroupCfg(
            terms={"value": ObservationTermCfg(func=fun)}, concatenate_terms=True, enable_corruption=False
        )

    cfg.observations = {
        "kv_tokens": visual,
        "q_cls": group(cached_cls),
        **cfg.observations,
        "q_task_cmd": group(goal_pose),
    }
    return cfg


class NativeCarryActor(ExtractorSonicAdapterModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("native_hoi_version", torch.tensor(1))
        self.register_buffer("hoi_reference_version", torch.tensor(2))

    def _decode(self, tokens, obs, cond=None):
        mean = super()._decode(tokens, obs, cond)
        target, blend = obs["hoi_reference_pd"].chunk(2, dim=-1)
        return torch.lerp(mean, target, blend)


def native_agent(checkpoint):
    cfg = asdict(adapt_sonic_ext_agent_cfg("hoi_vibe_uolm_matched"))
    actor = cfg["actor"]
    actor.update(
        class_name="hoi_native_vibe.NativeCarryActor",
        base_checkpoint=str(checkpoint),
        adapter_obs_group=["augmentation", "hoi", "kv_tokens"],
    )
    for spec in actor["extractor_cfg"].values():
        spec["class_name"] = NativePpoFusion.__module__ + "." + NativePpoFusion.__name__
    cfg["obs_groups"]["critic"] = ["policy", "augmentation", "hoi"]
    return cfg


def segmented_returns(algorithm, obs):
    """Bootstrap each 24-step segment independently before one pooled PPO update."""
    storage = algorithm.storage
    assert not algorithm.critic.is_recurrent and not algorithm.normalize_advantage_per_mini_batch
    n = storage.num_envs
    values, rewards, dones, returns = [
        x.view(-1, 24, n, 1) for x in (storage.values, storage.rewards, storage.dones, storage.returns)
    ]
    bootstrap = torch.cat((values[1:, 0], algorithm.critic(obs).detach().unsqueeze(0)))
    advantage = 0
    for t in reversed(range(24)):
        live = 1.0 - dones[:, t].float()
        delta = rewards[:, t] + live * algorithm.gamma * bootstrap - values[:, t]
        advantage = delta + live * algorithm.gamma * algorithm.lam * advantage
        returns[:, t] = values[:, t] + advantage
        bootstrap = values[:, t]
    adv = storage.returns - storage.values
    storage.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
