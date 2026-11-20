"""Frozen SONIC G1/TELEOP mode selection and original teacher."""
from da2_vibe.paths import load_checkpoint
import hashlib, inspect
from pathlib import Path
import torch
from rsl_rl.models.sonic_base_model import SonicBaseModel, _mlp, fsq_quantize, TOKEN_CACHE_KEY

class DualEncoderMixin:
    def _add_teleop(self, checkpoint):
        state = load_checkpoint(checkpoint, map_location="cpu", weights_only=False)
        self.teleop_encoder = _mlp(267, (2048, 1024, 512, 512), 64, "SiLU")
        self.teleop_encoder.load_state_dict(state["state_dict"], strict=True)
        self.teleop_encoder.eval().requires_grad_(False)
        self.teleop_weight_sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()

    def train(self, mode=True):
        result = super().train(mode)
        if hasattr(self, "teleop_encoder"):
            self.teleop_encoder.eval()
        return result

    def encode_tokens(self, obs, latent_residual=None, cond=None):
        if self.cache_tokens and TOKEN_CACHE_KEY in obs.keys():
            tokens = obs[TOKEN_CACHE_KEY]
        else:
            g1 = super().encode_tokens(obs, None, cond)
            z = self.teleop_encoder(obs["teleop_reference"])
            z = z.reshape(*z.shape[:-1], self.num_tokens, self.token_dim)
            teleop = fsq_quantize(z, self.fsq_levels).flatten(-2)
            tokens = torch.where(obs["reference_mode"] > 0.5, teleop, g1)
            self._last_tokens = tokens
        return tokens if latent_residual is None else tokens + latent_residual

class OriginalTeleopTeacher(DualEncoderMixin, SonicBaseModel):
    def __init__(self, *args, teleop_checkpoint, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_teleop(teleop_checkpoint)

def build_teacher(obs, actor_cfg, obs_groups, checkpoint, device):
    allowed = inspect.signature(SonicBaseModel.__init__).parameters
    kwargs = {k: v for k, v in actor_cfg.items() if k in allowed}
    kwargs.update(
        base_checkpoint=checkpoint, freeze_base=True, cache_tokens=False, distribution_cfg=None
    )
    teacher = (
        OriginalTeleopTeacher(
            obs, obs_groups, "actor", 29, teleop_checkpoint=actor_cfg["teleop_checkpoint"], **kwargs
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    digest = hashlib.sha256(
        Path(checkpoint).read_bytes() + Path(actor_cfg["teleop_checkpoint"]).read_bytes()
    ).hexdigest()
    return teacher, digest
