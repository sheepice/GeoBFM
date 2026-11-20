"""Contact-based Revo2 pickup and learned-policy locomotion demo."""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from da2_vibe.paths import load_settings
from .revo2_scene import make_scene
from .native_teleop import NativeTeleop


def run(args, configuration, output):
    import imageio.v2 as iio
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib

    torch.set_num_threads(2)
    torch.manual_seed(261115)
    settings = load_settings(configuration['policy_settings'])
    spec, model, mapping = make_scene(settings,
        None if args.no_hands else configuration['revo2_description'],
        stairs_height=args.height, stairs_start=configuration['stairs_start_m'],
        table=not args.no_table, object_position=configuration['object_position_m'],
        object_mass=configuration['object_mass_kg'], table_center=configuration.get('table_center_m', [.43, -.54, .79]))
    (output / 'scene.xml').write_text(spec.to_xml())
    mujoco.mj_saveModel(model, str(output / 'scene.mjb'))
    (output / 'configuration.json').write_text(json.dumps(dict(args=vars(args), demo=configuration, policy=settings), indent=2))
    data = mujoco.MjData(model)
    policy = NativeTeleop(settings, configuration['checkpoint'], model, mapping)
    initial = policy.pool.poses[0, policy.pool.cursor].copy()
    data.qpos[:7] = initial[:7]
    data.qpos[:2] = [0, 0]
    data.qpos[mapping['qadr']] = initial[7:]
    data.ctrl[mapping['aids']] = initial[7:]
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 112, 112)
    video_renderer = None if args.no_video else mujoco.Renderer(model, 720, 1280)
    option = mujoco.MjvOption()
    option.geomgroup[3] = False
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135, -22, 4.5
    font_path = Path(matplotlib.__file__).parent / 'mpl-data/fonts/ttf/DejaVuSans.ttf'
    font = ImageFont.truetype(str(font_path), 23)
    writer = None if args.no_video else iio.get_writer(output / 'demo.mp4', fps=25, codec='libx264', quality=8,
        macro_block_size=1, ffmpeg_params=['-movflags', '+faststart'])
    trace, poses = [], []
    speed = .4 if args.mode == 'walk' else 0.
    previous_task_target = None
    plan = None
    if args.mode == 'demo':
        from .pick_plan import PickPlan
        plan = PickPlan(model, data, mapping, configuration)
    try:
        for step in range(round(args.duration / .02) + 1):
            if step % 5 == 0:
                renderer.update_scene(data, camera='head_cam', scene_option=option)
                rgb = renderer.render().copy()
            detail = {}
            points = orientations = None
            if plan:
                speed, points, orientations, detail = plan.update(data, step*.02, *policy.default_targets(speed))
            target = policy.act(data, rgb, speed, points=points, orientations=orientations)
            if args.arm_servo and plan and plan.phase not in ('SETTLE', 'OBJECT_DROPPED', 'GRASP_FAILED', 'REACH_TIMEOUT'):
                from scipy.spatial.transform import Rotation
                position = np.asarray(detail['grasp_position'])
                task_target = np.asarray(detail['target'])
                desired_velocity = np.zeros(3) if previous_task_target is None else (task_target-previous_task_target)/.02
                previous_task_target = task_target.copy()
                jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jp, jr, position, plan.wrist)
                rotation = Rotation.from_matrix(data.xmat[plan.wrist].reshape(3,3))
                force = 400*(task_target-position) + 18*(desired_velocity-jp@data.qvel)
                moment = 16*rotation.inv().as_rotvec() - 1.2*(jr@data.qvel)
                tau = jp.T@force + jr.T@moment
                arm = np.arange(22,29)
                correction = np.clip(tau[mapping['vadr'][arm]] / model.actuator_gainprm[mapping['aids'][arm],0],-.35,.35)
                target[arm] += correction
                policy.builder.accept_action((target-mapping['default'])/mapping['scale'])
                detail['arm_servo_max_correction_rad'] = float(np.abs(correction).max())
            data.ctrl[mapping['aids']] = target
            quat = data.qpos[3:7]
            gravity = 2 * (quat[1]**2 + quat[2]**2) - 1
            row = dict(t=step*.02, root=data.qpos[:3].tolist(), gravity_z=float(gravity), speed_command=speed, **detail)
            trace.append(row)
            poses.append(data.qpos.copy())
            if step % 50 == 0:
                print(json.dumps(row), flush=True)
            if writer and step % 2 == 0:
                camera.lookat[:] = [.6 + .65*data.qpos[0], -.15, .85 + max(0, data.qpos[2]-.76)*.5]
                video_renderer.update_scene(data, camera=camera, scene_option=option)
                frame = Image.fromarray(video_renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 1280, 45), fill=(16, 22, 32))
                stage = detail.get('phase',args.mode.upper())
                draw.text((16, 9), f'Revo2 + DA2 TELEOP model_{policy.iteration} | {stage} | t={step*.02:.2f}s',font=font, fill='white')
                writer.append_data(np.asarray(frame))
                if step % 100 == 0:
                    frame.save(output / f'frame_{step:04d}.jpg')
            if gravity > -.3 or data.qpos[2] < .3:
                print('FALL', flush=True)
                break
            if plan and plan.phase in ('REACH_TIMEOUT', 'GRASP_FAILED', 'OBJECT_DROPPED') and step*.02-plan.since>.5:
                break
            for _ in range(round(.02/model.opt.timestep)):
                mujoco.mj_step(model, data)
    finally:
        if writer:
            writer.close()
        renderer.close()
        if video_renderer:
            video_renderer.close()
        (output / 'trace.json').write_text(json.dumps(trace, indent=2))
        np.savez_compressed(output / 'trajectory.npz', qpos=np.asarray(poses))
    print('RUN_FINISHED', flush=True)
    return 0
