"""Execute the exported RGB encoder + ONNX policy in MuJoCo using bundle optics."""
import argparse
import json
from pathlib import Path
from da2_vibe.paths import remap_metadata

from ..gpu_guard import verify_guarded_worker


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seconds', type=float, default=5.)
    args = p.parse_args()
    verify_guarded_worker()
    import mjlab
    import numpy as np
    import torch
    from .runtime import ExportedPolicy
    from .export import build_export_env
    from .camera import camera_contract, assert_sim_camera
    torch.set_num_threads(2)
    runtime = ExportedPolicy(args.bundle)
    env = build_export_env(remap_metadata(runtime.manifest['settings']))
    try:
        actual = camera_contract(next(s for s in env.unwrapped.cfg.scene.sensors if s.name == 'head_cam'))
        assert actual == runtime.manifest['camera'], 'Sim2sim camera differs from exported camera'
        assert_sim_camera(env.unwrapped, actual)
        obs, _ = env.reset()
        finite = True
        episodes = 0
        with torch.inference_mode():
            for step in range(round(args.seconds / env.unwrapped.step_dt)):
                rgb = env.unwrapped.scene['head_cam'].data.rgb[0].cpu().numpy()
                inputs = {k: obs[k].cpu().numpy() for k in runtime.manifest['input_shapes'] if k != 'img_tokens'}
                actions = runtime(rgb, inputs)
                finite &= bool(np.isfinite(actions).all())
                if not finite:
                    raise RuntimeError('Nonfinite exported action')
                obs, _, done, _ = env.step(torch.as_tensor(actions, device=env.device))
                if done.any():
                    runtime.reset()
                    episodes += 1
        report = dict(bundle=args.bundle, steps=step+1, seconds=args.seconds,
            camera=actual, compiled_camera_verified=True, finite_actions=finite,
            completed_episodes=episodes, hardware_test=False,
            scope='Exported RGB-to-action execution with matched optics; not a locomotion success benchmark')
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
