"""Training and evaluation use the same ablated visual observation."""
import torch

from ..da2_locomotion.observation import FrozenVisualTokens


def transform_tokens(tokens, mode, frames=2):
    if mode == 'normal':
        return tokens
    if mode == 'constant_tokens':
        return torch.zeros_like(tokens)
    if mode != 'repeat_current' or frames < 1 or tokens.shape[1] % frames:
        raise ValueError('Invalid visual-history ablation')
    current = tokens[:, -(tokens.shape[1] // frames):]
    return current.repeat(1, frames, 1)


class AblationDA2Tokens(FrozenVisualTokens):
    def __init__(self, cfg, env):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            super().__init__(cfg, env)

    @torch.no_grad()
    def __call__(self, env, **kwargs):
        mode = self.settings['ablation_input']
        if mode == 'constant_tokens':
            # Preserve the same policy tensor shape, without reading the current
            # camera or encoding scene-specific information. The critic is unchanged.
            return torch.zeros_like(self.history.values.flatten(1, 2))
        return transform_tokens(super().__call__(env, **kwargs), mode, self.settings['frames'])


def audit_blind_observations(raw, obs):
    """Check the actual collected inputs, without requesting an image."""
    visual = raw.da2_visual
    tokens = obs['kv_tokens']['img_tokens']
    if visual.settings['ablation_input'] != 'constant_tokens':
        raise RuntimeError('Blind input audit needs the constant-token observation')
    if tuple(tokens.shape[1:]) != (128, 96) or torch.count_nonzero(tokens).item():
        raise RuntimeError('Blind actor received nonconstant or malformed visual tokens')
    if visual.calls or visual.images:
        raise RuntimeError('Blind observation executed the image encoder')
    return dict(all_visual_tokens_zero=True, token_shape=list(tokens.shape),
                encoder_calls=visual.calls, encoded_images=visual.images,
                geometry_targets_used_for_supervision=False)


def reset_with_ablation_audit(env, settings, record):
    """Audit actual reset inputs independently of geometry replay sampling."""
    obs, extras = env.reset()
    if settings.get('ablation', {}).get('variant') == 'blind':
        # A resumed physics step need not be a geometry sampling step, so
        # collector.pending may be empty. Reuse reset's observation without
        # another observation-manager call or any extra corruption RNG draws.
        record['blind_input_audit'] = audit_blind_observations(env.unwrapped, obs)
    return obs, extras
