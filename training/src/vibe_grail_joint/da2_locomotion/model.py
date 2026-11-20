"""Multi-query/multi-head visual fusion on the actual SONIC adapter path."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from rsl_rl.modules.normalization import EmpiricalNormalization
from ..velocity.policy import CommandSonicActor


class VisualQueryExtractor(nn.Module):
    def __init__(self, query_dims, frames=2, patch_grid=(8, 8), num_heads=8,
                 attn_dim=64, latent_dim=128, query_mode='state', **unused):
        super().__init__()
        if unused:
            raise ValueError(f'Unknown extractor options: {sorted(unused)}')
        if attn_dim % num_heads or frames < 1:
            raise ValueError('Invalid attention dimensions/history')
        if query_mode not in ('state', 'constant'):
            raise ValueError('Unknown visual query mode')
        self.query_mode = query_mode
        self.query_groups = list(query_dims)
        self.frames, self.patch_grid = frames, tuple(patch_grid)
        self.num_heads, self.attn_dim, self.latent_dim = num_heads, attn_dim, latent_dim
        self.token_dim = 96
        self.token_norm = nn.LayerNorm(96)
        self.query_normalizers = nn.ModuleDict({k: EmpiricalNormalization(v) for k, v in query_dims.items()})
        self.w_q = nn.ModuleDict({k: nn.Linear(v, attn_dim) for k, v in query_dims.items()})
        self.w_k, self.w_v = nn.Linear(96, attn_dim), nn.Linear(96, attn_dim)
        self.frame_embedding = nn.Parameter(torch.zeros(frames, 96))
        nn.init.normal_(self.frame_embedding, std=.01)
        self.output = nn.Sequential(nn.Linear(len(query_dims) * attn_dim, latent_dim), nn.LayerNorm(latent_dim), nn.ELU())
        if query_mode == 'constant':
            # The two existing projection biases become learned query vectors.
            # Keep common parameter initialization/RNG exactly unchanged, but do
            # not count the unused state projection matrices as trainable capacity.
            for projection in self.w_q.values():
                projection.weight.requires_grad_(False)
        self.input_groups = ('kv_tokens', *self.query_groups)
        self.last_attn = None

    @classmethod
    def from_obs(cls, obs, group, query_groups, **kwargs):
        if obs[group]['img_tokens'].shape[-1] != 96:
            raise ValueError('Expected 96-dimensional compressed visual tokens')
        result = cls({q: obs[q].shape[-1] for q in query_groups}, **kwargs)
        result.input_groups = (group, *query_groups)
        return result

    def attention(self, tokens, *queries):
        batch, count, width = tokens.shape
        patches = math.prod(self.patch_grid)
        if count != self.frames * patches or width != 96:
            raise ValueError('Wrong visual history/grid dimensions')
        # Normalize the exact features consumed here; no statistics from a different backbone.
        t = self.token_norm(tokens.float()).reshape(batch, self.frames, patches, 96)
        t = (t + self.frame_embedding[None, :, None]).flatten(1, 2)
        q = torch.stack([self.w_q[name](torch.zeros_like(value) if self.query_mode == 'constant'
                                      else self.query_normalizers[name](value))
                         for name, value in zip(self.query_groups, queries, strict=True)], 1)
        head_dim = self.attn_dim // self.num_heads
        q = q.reshape(batch, len(self.query_groups), self.num_heads, head_dim).transpose(1, 2)
        k = self.w_k(t).reshape(batch, count, self.num_heads, head_dim).transpose(1, 2)
        v = self.w_v(t).reshape(batch, count, self.num_heads, head_dim).transpose(1, 2)
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(head_dim), -1)
        fused = (weights @ v).transpose(1, 2).reshape(batch, -1)
        return self.output(fused), weights

    def forward(self, token_td, *queries):
        latent, weights = self.attention(token_td['img_tokens'], *queries)
        self.last_attn = weights.detach()
        return latent

    @torch.no_grad()
    def update_normalization(self, token_td, *queries):
        if self.query_mode == 'constant':
            return
        for name, value in zip(self.query_groups, queries, strict=True):
            self.query_normalizers[name].update(value)

    @torch.no_grad()
    def metrics(self, **kwargs):
        if self.last_attn is None:
            return {}
        w = self.last_attn
        return {'DA2Attention/entropy': float(-(w * w.clamp_min(1e-9).log()).sum(-1).mean() / math.log(w.shape[-1])),
                'DA2Attention/max_probability': float(w.amax(-1).mean())}


class TerrainConditionedActor(CommandSonicActor):
    def __init__(self, *args, geometry_aux, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry_aux = dict(geometry_aux)
        # Auxiliary geometry uses the SAME fused visual representation as the action adapter.
        self.terrain_head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 256), nn.ELU(),
                                          nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 99))

    def auxiliary_loss(self, batch):
        prediction = self.terrain_head(self.encode(batch['obs']))
        target, valid = batch['height'], batch['valid'].float()
        error = prediction - target
        loss = (error.square() * valid).sum() / valid.sum().clamp_min(1)
        mae = (error.detach().abs() * valid).sum() / valid.sum().clamp_min(1)
        return loss, mae * self.geometry_aux['height_scale_m']
