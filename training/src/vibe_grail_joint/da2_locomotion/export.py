"""Export an RGB encoder (TorchScript) and stateless action policy (ONNX)."""
from da2_vibe.paths import load_checkpoint
from da2_vibe.paths import load_settings
import argparse
import copy
import json
from pathlib import Path

from ..gpu_guard import verify_guarded_worker


def build_export_env(settings):
    if settings.get('task_mode') in ('cylinder_stones_only_v1', 'cylinder_stones_flatmix_v2', 'cylinder_stones_sideaware_flatmix_v3'):
        from mjlab.rl import RslRlVecEnvWrapper
        from ..da2_stones.task import env_config, StoneEnv
        from orcs.core.robustness import strip_domain
        cfg = env_config(settings['schema_bank'], settings, 1)
        strip_domain(cfg)
        for group in cfg.observations.values():
            group.enable_corruption = False
        return RslRlVecEnvWrapper(StoneEnv(cfg, device='cuda:0', settings=settings))
    from .scenes import evaluation_env
    return evaluation_env(settings, seed=291030, duration=10., heights=(.2,),
                          speeds=(.4,), approaches=(0.,), limit_cases=1)[0]


def build_export_actor(settings, obs, checkpoint, raw):
    if settings.get('task_mode') not in ('cylinder_stones_only_v1', 'cylinder_stones_flatmix_v2', 'cylinder_stones_sideaware_flatmix_v3'):
        from .evaluate import load_actor
        return load_actor(settings, obs, checkpoint, raw)
    import torch
    from rsl_rl.utils import resolve_callable
    from ..da2_stones.task import agent_config
    agent = agent_config(settings['schema_bank'], settings, 1)
    cfg = copy.deepcopy(agent['actor'])
    cls = resolve_callable(cfg.pop('class_name'))
    actor = cls(obs=obs, obs_groups=agent['obs_groups'], obs_set='actor', output_dim=29, **cfg).to(raw.device)
    state = load_checkpoint(checkpoint, map_location=raw.device, weights_only=False)
    actor.load_state_dict(state['actor_state_dict'], strict=True)
    return actor.eval(), int(state['iter'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    verify_guarded_worker()
    import numpy as np
    import torch
    from torch import nn
    from tensordict import TensorDict
    import onnx
    import onnxruntime as ort
    from .encoder import file_sha256
    torch.set_num_threads(2)
    settings = load_settings(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    env = build_export_env(settings)
    try:
        obs, _ = env.reset()
        actor, iteration = build_export_actor(settings, obs, args.checkpoint, env.unwrapped)
        from .camera import camera_contract, assert_sim_camera
        camera = assert_sim_camera(env.unwrapped, camera_contract(next(s for s in env.unwrapped.cfg.scene.sensors if s.name == 'head_cam')))
        saved = load_checkpoint(args.checkpoint, map_location='cpu', weights_only=False)['infos']
        if saved['da2_locomotion']['camera'] != settings['camera']:
            raise ValueError('Export optics differ from the checkpoint training camera')
        (out/'camera.json').write_text(json.dumps(camera, indent=2))
        port_names = ('policy', 'tokenizer', 'augmentation', 'velocity_command', 'q_proprio', 'img_tokens')

        class Policy(nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = copy.deepcopy(actor).cpu().eval()
                self.actor._latest_update_obs = None

            def forward(self, policy, tokenizer, augmentation, velocity_command, q_proprio, img_tokens):
                observations = TensorDict(dict(policy=policy, tokenizer=tokenizer,
                    augmentation=augmentation, velocity_command=velocity_command, q_proprio=q_proprio,
                    kv_tokens=TensorDict({'img_tokens': img_tokens}, batch_size=[policy.shape[0]])),
                    batch_size=[policy.shape[0]])
                return self.actor(observations, stochastic_output=False)

        wrapper = Policy(actor)
        inputs = tuple((obs['kv_tokens']['img_tokens'] if name == 'img_tokens' else obs[name]).cpu()
                       for name in port_names)
        with torch.no_grad():
            expected = wrapper(*inputs).numpy()
            torch.onnx.export(wrapper, inputs, str(out/'policy.onnx'), input_names=list(port_names),
                              output_names=['actions'], opset_version=17, dynamo=False)
        onnx.checker.check_model(str(out/'policy.onnx'))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(out/'policy.onnx'), options, providers=['CPUExecutionProvider'])
        feeds = {name: value.numpy() for name, value in zip(port_names, inputs)}
        exported = session.run(None, feeds)[0]
        error = float(np.max(np.abs(expected-exported)))
        if error > 2e-4:
            raise RuntimeError(f'ONNX action parity failed: {error}')
        encoder = env.unwrapped.da2_visual.encoder
        rgb = env.unwrapped.scene['head_cam'].data.rgb[:1].clone()
        with torch.no_grad():
            traced = torch.jit.trace(encoder, (rgb,), check_trace=True)
            traced.save(str(out/'encoder.ts'))
            loaded = torch.jit.load(str(out/'encoder.ts'), map_location=env.device)
            feature_error = float((loaded(rgb).float()-encoder(rgb).float()).abs().max())
        if feature_error > 1e-3:
            raise RuntimeError(f'TorchScript feature parity failed: {feature_error}')
        manifest = dict(version=1, checkpoint=str(Path(args.checkpoint).resolve()), iteration=iteration,
                        checkpoint_sha256=file_sha256(args.checkpoint), settings=settings, camera=camera,
                        camera_sha256=file_sha256(out/'camera.json'),
                        policy_sha256=file_sha256(out/'policy.onnx'), encoder_sha256=file_sha256(out/'encoder.ts'),
                        input_shapes={name:list(value.shape) for name,value in zip(port_names,inputs)},
                        action_max_abs_error=error, encoder_max_abs_error=feature_error,
                        auxiliary_heads_removed=True, visual_history='oldest_to_current',
                        runtime='encoder.ts on CUDA with RGB uint8 NHWC; policy.onnx accepts cached visual history',
                        visual_update_interval_steps=settings['visual']['update_interval_steps'])
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        np.savez(out/'parity_inputs.npz', **feeds, expected_action=expected)
        print(json.dumps(manifest,indent=2),flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
