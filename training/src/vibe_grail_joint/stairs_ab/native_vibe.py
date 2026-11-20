"""Author ViBe actor/PPO on our stair task; no added command or auxiliary loss."""
import copy
from dataclasses import asdict, replace
import torch
from torch.utils.checkpoint import checkpoint

from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from rsl_rl.models import ExtractorSonicAdapterModel
from ..velocity.policy import CommandSonicActor

from . import task as matched
from .theia import TheiaTokens, cached_cls
from .theia import NativeTheiaFusion


class NativePpoFusion(NativeTheiaFusion):
    @torch.no_grad()
    def metrics(self, max_samples=4096):
        # Logging-only RankMe SVD must not allocate a late cuSOLVER context
        # after a full-size PPO update. Actor forward/backward stay on CUDA.
        original = self.last_z
        try:
            self.last_z = None if original is None else original[:max_samples].float().cpu()
            return super().metrics(max_samples)
        finally:
            self.last_z = original


class NativeSonicActor(ExtractorSonicAdapterModel):
    """Native math/parameters; optional activation recomputation during PPO."""
    sonic_action = CommandSonicActor.sonic_action

    def __init__(self, *args, activation_checkpointing=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation_checkpointing = activation_checkpointing

    def _adapter_cond(self, obs):
        if self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(super()._adapter_cond, obs, use_reentrant=False)
        return super()._adapter_cond(obs)

    def _decode(self, tokens, obs, cond=None):
        if not self.activation_checkpointing or not torch.is_grad_enabled() or not self.adapt_decoder:
            return super()._decode(tokens, obs, cond)
        base_input = torch.cat([tokens, self._get_proprio(obs)], dim=-1)
        cond = self._adapter_cond(obs) if cond is None else cond
        decoder = self.decoder
        def first(x, conditioning):
            value = decoder.base_linears[0](x)
            if decoder.adapters[0] is not None:
                value = value + decoder.adapters[0](conditioning)
            return value
        h = checkpoint(first, base_input, cond, use_reentrant=False)
        for i in range(1, len(decoder.base_linears)):
            def block(value, index=i):
                value = decoder.activation(value)
                result = decoder.base_linears[index](value)
                if decoder.adapters[index] is not None:
                    result = result + decoder.adapters[index](value)
                return result
            h = checkpoint(block, h, use_reentrant=False)
        return h


def configure_native_observations(cfg, settings):
    from vibe.tasks.perloco.config.g1.observation_cfgs import ObsCtx, attach_ext_obs
    camera = settings['camera']
    cfg.scene.sensors = tuple(replace(s, **{k: tuple(camera[k]) if k in ('pos', 'quat') else camera[k]
        for k in ('width', 'height', 'fovy', 'parent_body', 'pos', 'quat')})
        if s.name == 'head_cam' else s for s in cfg.scene.sensors)
    attach_ext_obs(cfg, ctx=ObsCtx(p={'command_name': 'motion'}))
    cfg.observations['kv_tokens'].terms = {'img_tokens': ObservationTermCfg(
        func=TheiaTokens, params={'visual': copy.deepcopy(settings['visual'])})}
    cfg.observations['q_cls'] = ObservationGroupCfg(
        terms={'img_cls': ObservationTermCfg(func=cached_cls)},
        concatenate_terms=True, enable_corruption=False)
    return cfg


def scaffold(settings):
    value = copy.deepcopy(settings)
    value['comparison']['method'] = 'vibe_matched'
    value['camera'].update(width=112, height=112, fovy=42.5, fovx=42.5,
                           fov_policy='explicit_square_pinhole')
    return value


def env_config(bank_path, settings, num_envs):
    cfg = matched.env_config(bank_path, scaffold(settings), num_envs)
    return configure_native_observations(cfg, settings)


def agent_config(bank_path, settings, num_envs):
    if (not settings.get('native_vibe') or not settings['comparison'].get('stairs_only')
            or settings['initialization'] != 'raw_sonic_fresh_visual'
            or settings.get('dagger') is not None
            or settings['geometry_aux']['coefficient'] != 0
            or settings['flat_repair']['coefficient'] != 0
            or settings['posture_revision']['upper_preservation_coef'] != 0):
        raise ValueError('Native PPO requires fresh SONIC, stairs only and no additional learning objectives')
    return build_native_agent(settings, 'vibe_native_ppo_our_stairs')


def build_native_agent(settings, experiment_name):
    """Author actor/PPO recipe shared by task-specific validated adapters."""
    from vibe.tasks.perloco.config.g1.agent_cfgs import adapt_sonic_ext_agent_cfg
    cfg = asdict(adapt_sonic_ext_agent_cfg(experiment_name))
    cfg.update(seed=settings['seed'], max_iterations=settings['iterations'],
               save_interval=settings['save_interval'], logger='tensorboard', amp_dtype=None)
    cfg['actor']['base_checkpoint'] = settings['sonic_checkpoint']
    cfg['actor']['class_name'] = 'vibe_grail_joint.stairs_ab.native_vibe.NativeSonicActor'
    cfg['actor']['activation_checkpointing'] = True
    # Same floating-point normalization math; FP16 rollout token storage only.
    for value in cfg['actor']['extractor_cfg'].values():
        value['class_name'] = 'vibe_grail_joint.stairs_ab.native_vibe.NativePpoFusion'
    cfg['critic']['class_name'] = 'vibe_grail_joint.stairs_ab.initialization.MatchedCritic'
    cfg['critic']['initialization_seed'] = settings['seed'] + 800000
    assert cfg['algorithm']['class_name'] == 'PPO'
    assert cfg['algorithm']['num_learning_epochs'] == 5
    assert cfg['algorithm']['num_mini_batches'] == 4
    return cfg


def load_original(settings, obs, checkpoint, device):
    import hashlib
    from pathlib import Path
    from da2_vibe.paths import load_checkpoint
    from ..velocity.dagger_policy import build_teacher
    from vibe.tasks.perloco.config.g1.agent_cfgs import adapt_sonic_ext_agent_cfg
    cfg = asdict(adapt_sonic_ext_agent_cfg('original_baseline'))
    cfg['actor']['base_checkpoint'] = settings['sonic_checkpoint']
    baseline = settings['evaluation_baseline']
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if digest != baseline['checkpoint_sha256']:
        raise ValueError('Original baseline checkpoint identity mismatch')
    if baseline['method'] == 'sonic_original':
        teacher, _ = build_teacher(obs, cfg['actor'], cfg['obs_groups'], checkpoint, device)
        return teacher, None
    if baseline['method'] != 'vibe_original':
        raise ValueError('Unknown original baseline')
    acfg = dict(cfg['actor']); acfg.pop('class_name')
    actor = ExtractorSonicAdapterModel(obs=obs, obs_groups=cfg['obs_groups'],
        obs_set='actor', output_dim=29, **acfg).to(device).eval()
    state = load_checkpoint(checkpoint, map_location=device, weights_only=False)
    actor.load_state_dict(state['actor_state_dict'], strict=True)
    actor.requires_grad_(False)
    return actor, int(state['iter'])
