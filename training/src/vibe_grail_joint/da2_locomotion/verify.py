"""Bounded real-weight validation; no training or production artifact is changed."""
from da2_vibe.paths import load_settings
import argparse
import json
from pathlib import Path
import time

from ..gpu_guard import verify_guarded_worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bundle')
    args = parser.parse_args()
    verify_guarded_worker()
    import torch
    from .encoder import FrozenVisualEncoder
    from .model import VisualQueryExtractor
    torch.set_num_threads(2)
    settings = load_settings(args.config)
    encoder = FrozenVisualEncoder(settings['visual'], 'cuda:0')
    images = torch.randint(0, 256, (4, 112, 112, 3), dtype=torch.uint8, device='cuda:0')
    started = time.monotonic()
    features = encoder(images)
    changed = images.clone()
    changed[1].zero_()
    changed_features = encoder(changed)
    torch.testing.assert_close(features[0], changed_features[0], atol=0, rtol=0)
    assert not torch.equal(features[1], changed_features[1])
    visual = settings['visual']
    extractor = VisualQueryExtractor({'q_proprio': 93, 'velocity_command': 8}, frames=visual['frames'],
                                     patch_grid=visual['patch_grid'], num_heads=visual['num_heads']).cuda()
    tokens = features.repeat(1, visual['frames'], 1).detach()
    latent, attention = extractor.attention(tokens, torch.randn(4, 93, device='cuda'),
                                           torch.randn(4, 8, device='cuda'))
    target = torch.randn_like(latent)
    loss = (latent-target).square().mean()
    loss.backward()
    assert extractor.w_v.weight.grad.abs().sum() > 0
    encoder.assert_frozen()
    report = dict(status='passed', feature_shape=list(features.shape),
                  finite=bool(torch.isfinite(features).all()), encoder_frozen=True,
                  independent_batch_images=True, image_intervention_changes_features=True,
                  action_path_fusion_gradient_norm=float(extractor.w_v.weight.grad.norm()),
                  attention_shape=list(attention.shape), seconds=time.monotonic()-started,
                  encoder_backend=settings['visual']['backend'],
                  encoder_checkpoint_sha256=settings['visual'].get('checkpoint_sha256'),
                  theia_revision=settings['visual'].get('theia_revision'),
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    if args.bundle:
        import numpy as np
        from .runtime import ExportedPolicy
        runtime = ExportedPolicy(args.bundle)
        data = np.load(Path(args.bundle)/'parity_inputs.npz')
        observations = {name: data[name] for name in runtime.manifest['input_shapes'] if name != 'img_tokens'}
        action = runtime(images[0].cpu().numpy(), observations)
        if not np.isfinite(action).all() or action.shape != (1, 29):
            raise RuntimeError('Exported RGB-to-action runtime failed')
        runtime.reset()
        repeated = runtime(images[0].cpu().numpy(), observations)
        # CUDA TorchScript can fuse BF16 operations after its profiling invocation.
        # Bound the resulting action error rather than promise bitwise equality.
        np.testing.assert_allclose(action, repeated, rtol=1e-5, atol=2e-4)
        report['exported_rgb_to_action_runtime'] = 'passed; finite actions and reset repeatability within 2e-4'
        report['runtime_reset_max_action_error'] = float(np.max(np.abs(action-repeated)))
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
