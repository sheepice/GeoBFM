"""Condition the DA2/SONIC action decoder on measured and commanded box state."""

import torch
from torch import nn
from vibe_grail_joint.da2_locomotion.model import TerrainConditionedActor


class CarryActor(TerrainConditionedActor):
    def __init__(self, obs, *args, **kwargs):
        super().__init__(obs, *args, **kwargs)
        self.requires_grad_(False)
        self.decoder.adapters.requires_grad_(True)
        width = self.decoder.adapters[0].down.in_features
        self.hoi_encoder = nn.Sequential(
            nn.Linear(obs["hoi"].shape[-1], 128), nn.ELU(), nn.Linear(128, width)
        )
        nn.init.zeros_(self.hoi_encoder[-1].weight)
        nn.init.zeros_(self.hoi_encoder[-1].bias)

    def _adapter_cond(self, obs):
        return super()._adapter_cond(obs) + self.hoi_encoder(obs["hoi"])

    def update_normalization(self, obs):
        # Preserve the visual and twist normalizers of the pretrained actor.
        pass


class ReferenceCarryActor(CarryActor):
    """Fresh SONIC adapters with joint-specific HOI references in the PD mean."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extractors.requires_grad_(True)
        self.command_encoder.requires_grad_(True)
        self.register_buffer("hoi_reference_version", torch.tensor(1))

    def _decode(self, tokens, obs, cond=None):
        policy = super()._decode(tokens, obs, cond)
        target, blend = obs["hoi_reference_pd"].chunk(2, dim=-1)
        return torch.lerp(policy, target, blend)

    def update_normalization(self, obs):
        # Update only adapter-stream statistics. SONIC's tokenizer/proprio stay frozen.
        from rsl_rl.models.extractor_model import ExtractedAdapterStreamMixin

        ExtractedAdapterStreamMixin.update_normalization(self, obs)
