"""Pinned frozen visual backbones, with no depth decoder in the control path."""
from __future__ import annotations
from da2_vibe.paths import load_checkpoint

import hashlib
import sys
from pathlib import Path

import torch
from torch import nn

DA2_BACKBONES = {'da2_metric_small': 'vits', 'da2_metric_base': 'vitb', 'da2_metric_large': 'vitl'}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def group_channels(tokens, groups=32):
    """Average channels without mixing image locations (B, P, D -> B, P, G)."""
    if tokens.shape[-1] % groups:
        raise ValueError('Channel groups must divide the backbone width')
    return tokens.reshape(*tokens.shape[:-1], groups, -1).mean(-1)


class FrozenVisualEncoder(nn.Module):
    def __init__(self, settings, device='cpu'):
        super().__init__()
        self.settings = dict(settings)
        self.backend = settings['backend']
        self.grid = tuple(settings['patch_grid'])
        self.compute_dtype = getattr(torch, settings.get('dtype', 'bfloat16'))
        if self.backend in DA2_BACKBONES:
            checkpoint = settings['checkpoint']
            if file_sha256(checkpoint) != settings['checkpoint_sha256']:
                raise ValueError('DA2 checkpoint SHA256 mismatch')
            source = Path(settings['source']) / 'metric_depth'
            sys.path.insert(0, str(source))
            from depth_anything_v2.dinov2 import DINOv2
            self.backbone = DINOv2(model_name=DA2_BACKBONES[self.backend])
            full = load_checkpoint(checkpoint, map_location='cpu', weights_only=True)
            weights = {k.removeprefix('pretrained.'): v for k, v in full.items()
                       if k.startswith('pretrained.')}
            self.backbone.load_state_dict(weights, strict=True)
            depth = len(self.backbone.blocks)
            self.layers = (depth // 3 - 1, 2 * depth // 3 - 1, depth - 1)
        elif self.backend == 'theia_tiny':
            from ..vision import FrozenTheiaFeature, THEIA_REVISION
            if settings.get('theia_revision') != THEIA_REVISION:
                raise ValueError('Theia ablation must pin the original frozen backbone revision')
            self.backbone = FrozenTheiaFeature._load(
                'theia', 'theia-tiny-patch16-224-cddsv', 'cpu', torch.float32)
            self.layers = ()
        else:
            raise ValueError(f'Unknown visual backend {self.backend}')
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.to(device=device, dtype=self.compute_dtype).requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, rgb):
        if rgb.shape[1:] != (112, 112, 3) or rgb.dtype != torch.uint8:
            raise ValueError('Encoder requires uncropped 112x112 RGB uint8 NHWC')
        x = rgb.permute(0, 3, 1, 2).to(dtype=self.mean.dtype, device=self.mean.device) / 255.
        x = (x - self.mean) / self.std
        if self.backend in DA2_BACKBONES:
            # GeoLoco extracts raw intermediate blocks, not the depth decoder or CLS.
            tokens = self.backbone.prepare_tokens_with_masks(x)
            features = []
            for index, block in enumerate(self.backbone.blocks):
                tokens = block(tokens)
                if index in self.layers:
                    features.append(group_channels(tokens[:, 1:], 32))
            output = torch.cat(features, -1)
        else:
            tokens = self.backbone.backbone.model(
                pixel_values=x, interpolate_pos_encoding=True).last_hidden_state[:, 1:]
            output = group_channels(tokens, 96)
        if output.shape[1:] != (self.grid[0] * self.grid[1], 96):
            raise RuntimeError(f'Unexpected dense feature shape {output.shape}')
        return output.to(torch.float16)

    def assert_frozen(self):
        if self.training or any(p.requires_grad or p.grad is not None for p in self.parameters()):
            raise RuntimeError('Frozen visual encoder was modified by training')
