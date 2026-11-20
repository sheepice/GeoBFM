#!/usr/bin/env python3
"""Check an actual frozen DA2 checkpoint and its fixed-width feature interface on CPU."""
import argparse
import json
from pathlib import Path
import _bootstrap
import torch
from da2_vibe.paths import load_settings
from vibe_grail_joint.da2_locomotion.encoder import FrozenVisualEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    settings = load_settings(args.config)['visual']
    settings['dtype'] = 'float32'
    encoder = FrozenVisualEncoder(settings)
    rgb = torch.randint(0, 256, (2, 112, 112, 3), dtype=torch.uint8)
    tokens = encoder(rgb)
    separate = torch.cat([encoder(x[None]) for x in rgb])
    encoder.assert_frozen()
    checks = dict(shape=tuple(tokens.shape) == (2, 64, 96),
                  finite=bool(torch.isfinite(tokens).all()),
                  responds_to_rgb=bool((tokens[0] - tokens[1]).abs().max() > 1e-3),
                  batch_independent=torch.allclose(tokens, separate, atol=1e-3, rtol=1e-3),
                  no_autograd=not tokens.requires_grad)
    result = dict(status='passed' if all(checks.values()) else 'failed', checks=checks,
                  backend=settings['backend'], sha256=settings['checkpoint_sha256'],
                  backbone_parameters=sum(p.numel() for p in encoder.backbone.parameters()),
                  backbone_width=encoder.backbone.embed_dim, output_shape=list(tokens.shape),
                  layers=list(encoder.layers), device='cpu', dtype='float32')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if result['status'] != 'passed': raise SystemExit(1)


if __name__ == '__main__': main()
