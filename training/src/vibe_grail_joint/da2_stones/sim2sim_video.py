"""Record exported encoder.ts + policy.onnx in the pillar physics environment."""
import argparse
import copy
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--level', type=int, default=9)
    p.add_argument('--variant', type=int, default=0)
    p.add_argument('--start', choices=('static', 'walking'), default='static')
    p.add_argument('--seed', type=int, default=271107)
    p.add_argument('--seconds', type=float, default=30.)
    p.add_argument('--reference-bank', help='Evaluation-only continuous reference bank; checkpoint is unchanged')
    p.add_argument('--approach-extra-m', type=float, default=0.)
    p.add_argument('--nominal-entry', action='store_true', help='Reset once to nominal stand, then policy controls every step')
    a = p.parse_args()
    from ..gpu_guard import verify_guarded_worker
    verify_guarded_worker()
    import mjlab
    import torch
    import numpy as np
    import imageio.v2 as iio
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    from mjlab.rl import RslRlVecEnvWrapper
    from orcs.core.robustness import strip_domain
    from ..da2_locomotion.runtime import ExportedPolicy
    from ..da2_locomotion.camera import camera_contract, assert_sim_camera
    from ..common.review_camera import add_review_camera
    from .mixed_command import MixedStoneCommand, MixedStoneCommandCfg
    from .task import env_config, StoneEnv
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    runtime = ExportedPolicy(a.bundle)
    settings = copy.deepcopy(runtime.manifest['settings'])
    if a.reference_bank:
        import hashlib
        settings['flat_training']['reference_bank'] = str(Path(a.reference_bank).resolve())
        settings['flat_training']['reference_sha256'] = hashlib.sha256(Path(a.reference_bank).read_bytes()).hexdigest()
    if not 0 <= a.approach_extra_m <= 6:
        raise ValueError('Approach extension must stay within 0..6 m atlas clearance')
    if not settings.get('flat_training'):
        raise ValueError('This recorder requires the stand/start-stop pillar recipe')
    if not 0 <= a.level < settings['stones']['levels'] or not 0 <= a.variant < settings['profile_variants']:
        raise ValueError('Invalid level/variant')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)

    class FixedCommand(MixedStoneCommand):
        def _reset_task(self, ids, clip_ids, time_steps, origins):
            super()._reset_task(ids, clip_ids, time_steps, origins)
            if a.nominal_entry:
                pos = self.tile_origins[self.tile[ids]].clone()
                pos[:, 2] = self.robot.data.default_root_state[ids, 2]
                self._write_reference_state_to_sim(ids, pos, self.robot.data.default_root_state[ids, 3:7],
                    torch.zeros_like(pos), torch.zeros_like(pos), self.robot.data.default_joint_pos[ids].clone(),
                    torch.zeros_like(self.robot.data.default_joint_pos[ids]))

        def _sample_init_frame(self, ids):
            chosen, frames = super()._sample_init_frame(ids)
            self.tile[ids] = a.level * self.cfg.profile_variants + a.variant
            self.phase[ids] = 0
            self.uses_clip[ids] = a.start == 'static'
            self.static_reset[ids] = a.start == 'static'
            if a.start == 'static':
                self.reference_id[ids] = 5
                self.reference_step[ids] = 0
                chosen[:] = self.flat_clip_start + 5
                frames = self.motion.clip_offsets[chosen]
                self.speed[ids] = 0.
            else:
                chosen[:] = self.replay_clips + 1
                frames = self.motion.clip_offsets[chosen] + self.pool.cursor
                self.speed[ids] = self.cfg.fixed_speed
            origin = self.tile_origins[self.tile[ids]].clone()
            origin[:, :2] -= self.motion.body_pos_w[frames, 0, :2]
            self._env.scene.env_origins[ids] = origin
            return chosen, frames

    @dataclass(kw_only=True)
    class FixedCfg(MixedStoneCommandCfg):
        def build(self, env):
            return FixedCommand(self, env)

    cfg = env_config(settings['schema_bank'], settings, 1)
    old = cfg.commands['motion']
    cfg.commands['motion'] = FixedCfg(**{f.name: getattr(old, f.name) for f in fields(FixedCfg)})
    tile = cfg.commands['motion'].random_tiles[a.level * settings['profile_variants'] + a.variant]
    for key in ('start','course_end','goal'):
        tile[key] += a.approach_extra_m
    for shape in (*tile['solids'], *tile.get('mats', [])):
        shape['x'] += a.approach_extra_m
    if a.approach_extra_m:
        # Only selected course is rendered/collidable: neighboring atlas courses
        # must not accidentally enter the camera during the long flat approach.
        cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs'].random_tiles = (tile,)
        def extended_stall(env):
            c = env.command_manager.get_term('motion')
            if not hasattr(env, 'guidance'):
                return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            env.guidance.measure()
            return (env.guidance.steps_since_progress > 500 + round(a.approach_extra_m / .6 / env.step_dt)) & (c.speed > .001)
        cfg.terminations['stone_stall'].func = extended_stall
    cfg.episode_length_s = max(35., a.seconds + 1.)
    add_review_camera(cfg, dict(tile, base=tile['height']))
    if a.nominal_entry:
        # A separate fixed close camera makes policy-entry posture visible.
        add_review_camera(cfg, dict(tile, goal=2.5, base=tile['height']))
        cfg.scene.sensors = (*cfg.scene.sensors[:-1], replace(cfg.scene.sensors[-1], name='entry_camera'))
    cfg.scene.sensors = tuple(replace(s, width=1280, height=720) if s.name == 'review_camera' else s
                              for s in cfg.scene.sensors)
    strip_domain(cfg)
    cfg.events.pop('policy_update_counter', None)
    for group in cfg.observations.values():
        group.enable_corruption = False
        for term in group.terms.values():
            term.noise = None

    class ObservedEnv(StoneEnv):
        terminal = None
        def _reset_idx(self, env_ids=None):
            if hasattr(self, 'guidance') and int(self.episode_length_buf[0]) > 0 and self.terminal is None:
                c = self.command_manager.get_term('motion')
                self.terminal = dict(strict_success=bool(c.terminal_success[0]),
                    task_failure=bool(c.terminal_fall[0]),
                    pillars=int(self.guidance.sequence.progress[0]),
                    causes=[name for name in cfg.terminations if bool(self.termination_manager.get_term(name)[0])],
                    gravity_z=float(c.robot.data.projected_gravity_b[0, 2]))
            return super()._reset_idx(env_ids)

    raw = ObservedEnv(cfg, device='cuda:0', settings=settings)
    env = RslRlVecEnvWrapper(raw)
    writer = None
    trace = []
    try:
        obs, _ = env.reset()
        raw.terminal = None
        camera = camera_contract(next(s for s in cfg.scene.sensors if s.name == 'head_cam'))
        assert camera == runtime.manifest['camera']
        assert_sim_camera(raw, camera)
        runtime.reset()
        c = raw.command_manager.get_term('motion')
        initial = dict(root=c.robot.data.root_link_pos_w[0].cpu().tolist(),
                       linear_velocity=c.robot.data.root_link_lin_vel_w[0].cpu().tolist(),
                       joint_velocity=c.robot_joint_vel[0].cpu().tolist(),
                       command_speed=float(c.speed[0]))
        initial_head = raw.scene['head_cam'].data.rgb[0].cpu().numpy()
        Image.fromarray(initial_head).save(out/'head_initial.png')
        def yellow_pixels(rgb):
            return int(((rgb[...,0]>100)&(rgb[...,1]>80)&(rgb[...,2]<80)&
                        (rgb[...,0].astype(float)>rgb[...,2]+40)&(rgb[...,1].astype(float)>rgb[...,2]+40)).sum())
        initial['yellow_pixels_in_head_rgb'] = yellow_pixels(initial_head)
        if a.nominal_entry and a.approach_extra_m and initial['yellow_pixels_in_head_rgb']:
            raise RuntimeError('Yellow course visible at policy entry; inspect head_initial.png before recording')
        font = ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'), 20)
        writer = iio.get_writer(out/'rollout.mp4', fps=25, codec='libx264', quality=8,
            macro_block_size=1, ffmpeg_params=['-threads','1','-preset','fast','-movflags','+faststart'])
        frames = 0
        for step in range(round(a.seconds / raw.step_dt)):
            with torch.inference_mode():
                rgb = raw.scene['head_cam'].data.rgb[0].cpu().numpy()
                speed = float(c.speed[0])
                progress = int(raw.guidance.sequence.progress[0])
                if step % 2 == 0:
                    close = a.nominal_entry and float(c.robot.data.root_link_pos_w[0,0]-c.tile_origins[c.tile[0],0]) < 2.
                    camera_name = 'entry_camera' if close else 'review_camera'
                    frame = Image.fromarray(raw.scene[camera_name].data.rgb[0].cpu().numpy().copy()).resize((1280,720))
                    frame.paste(Image.fromarray(rgb).resize((224,224)), (1046,60))
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0,0,1280,45), fill=(20,25,30))
                    draw.text((12,10), f'POLICY ACTIVE | model {runtime.manifest["iteration"]} | L{a.level} V{a.variant} | cmd {speed:.1f} m/s | t={step*.02:.2f}s | {progress}/9', font=font, fill='white')
                    writer.append_data(np.asarray(frame))
                    frames += 1
                    if step in (0, 124, 250, 500, 1000, 1498, 1500, 1750, 2000):
                        frame.save(out/f'frame_{step:04d}.jpg')
                        Image.fromarray(rgb).save(out/f'head_{step:04d}.png')
                inputs = {k: obs[k].cpu().numpy() for k in runtime.manifest['input_shapes'] if k != 'img_tokens'}
                actions = runtime(rgb, inputs)
                if not np.isfinite(actions).all():
                    raise RuntimeError('Nonfinite exported action')
                trace.append(dict(t=step*.02, speed=speed, pillars=progress,
                    root=(c.robot.data.root_link_pos_w[0]-c.tile_origins[c.tile[0]]).cpu().tolist(),
                    gravity_z=float(c.robot.data.projected_gravity_b[0,2]),
                    root_velocity=c.robot.data.root_link_lin_vel_w[0].cpu().tolist(),
                    yellow_pixels=yellow_pixels(rgb),
                    action_max=float(np.abs(actions).max())))
                obs, _, done, _ = env.step(torch.as_tensor(actions, device=raw.device))
                if bool(done[0]):
                    break
        report = dict(bundle=str(Path(a.bundle).resolve()), checkpoint=runtime.manifest['checkpoint'],
            iteration=runtime.manifest['iteration'], checkpoint_sha256=runtime.manifest['checkpoint_sha256'],
            level=a.level, variant=a.variant, start=a.start, seed=a.seed, tile=tile,
            approach_extra_m=a.approach_extra_m, nominal_entry=a.nominal_entry,
            evaluation_reference_bank=settings['flat_training']['reference_bank'],
            seconds=(step+1)*raw.step_dt, video_frames=frames, camera=camera, initial=initial,
            result=raw.terminal or dict(strict_success=False, timed_out=True, pillars=int(raw.guidance.sequence.progress[0])),
            protocol='Exported encoder.ts on GPU + policy.onnx on CPU; same MuJoCo/Warp physics backend as training; deterministic actions; no action blending; first episode only; pre-reset video frames',
            limitation='Simulation deployment-path check, not cross-physics-engine or real-robot validation')
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        (out/'trace.json').write_text(json.dumps(trace)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == '__main__':
    main()
