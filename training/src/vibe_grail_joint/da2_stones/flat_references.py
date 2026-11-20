"""Generate continuous SONIC idle/walk/start-stop references on CPU.

Planner commands change within one stream, preserving its generated context.
There is no splice between unrelated standing and walking poses. These files
contain reference motion, not expert physics rollouts; the frozen controller
labels student states during training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import mujoco
import numpy as np

from ..kinematics import angular_velocity
from ..velocity.planner import PlannerPool

NAMES = ('stand', 'walk03', 'walk06', 'start_stop03', 'start_stop06', 'pillar_start06')


def scheduled_speeds(t, pillar_hold_seconds=2.5):
    walking = (3. <= t < 9.) or (13. <= t < 19.)
    return np.array([0., .3, .6, .3 if walking else 0., .6 if walking else 0.,
                     .6 if t >= pillar_hold_seconds else 0.], dtype=np.float32)


def generate(planner, output, seed=261106, seconds=37., pillar_hold_seconds=2.5):
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from ..prepare import build_model
    cfg = get_g1_robot_cfg()
    model, bodies = build_model()
    pose = model.qpos0.copy().astype(np.float32)
    pose[:3], pose[3:7] = cfg.init_state.pos, cfg.init_state.rot
    joints = [model.joint(i).name for i in range(model.njnt)
              if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    for name in joints:
        for pattern, value in cfg.init_state.joint_pos.items():
            if re.fullmatch(pattern, name):
                pose[model.joint(name).qposadr[0]] = value
    pool = PlannerPool(planner, pose, scheduled_speeds(0), seed)
    count = round(seconds / .02)
    poses = np.empty((len(NAMES), count, 36), np.float32)
    commands = np.empty((len(NAMES), count), np.float32)
    translation = np.zeros((len(NAMES), 2), np.float32)
    for step in range(count):
        desired = scheduled_speeds(step * .02, pillar_hold_seconds)
        pool.speeds[:] = desired
        if step:
            _, recenter = pool.advance()
            translation += recenter
        poses[:, step] = pool.poses[:, pool.cursor]
        poses[:, step, :2] += translation
        commands[:, step] = desired
        if step % 250 == 0:
            print(f'generated {step}/{count} reference frames per stream', flush=True)
    # Per-clip local origin; preserve the continuous trajectory within each clip.
    poses[:, :, :2] -= poses[:, :1, :2].copy()
    body_ids = [model.body(n).id for n in bodies]
    data = mujoco.MjData(model)
    pos = np.empty((len(NAMES), count, len(bodies), 3), np.float32)
    quat = np.empty((len(NAMES), count, len(bodies), 4), np.float32)
    for stream in range(len(NAMES)):
        for frame, qpos in enumerate(poses[stream]):
            data.qpos[:] = qpos
            mujoco.mj_kinematics(model, data)
            pos[stream, frame], quat[stream, frame] = data.xpos[body_ids], data.xquat[body_ids]
    channels = dict(joint_pos=poses[:, :, 7:], joint_vel=np.gradient(poses[:, :, 7:], .02, axis=1),
                    body_pos_w=pos, body_quat_w=quat, body_lin_vel_w=np.gradient(pos, .02, axis=1),
                    body_ang_vel_w=np.stack([angular_velocity(q, .02) for q in quat]))
    if any(not np.isfinite(v).all() for v in channels.values()):
        raise RuntimeError('Nonfinite generated reference')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(version=1, names=NAMES, joint_names=joints, body_names=list(bodies), dt=.02,
                    seed=seed, frames_per_clip=count, pillar_hold_seconds=pillar_hold_seconds,
                    planner_sha256=hashlib.sha256(Path(planner).read_bytes()).hexdigest(),
                    reference_only=True, generated_by='continuous SONIC context with explicit mode 0/2')
    with output.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f, **channels, command_speed=commands, metadata=json.dumps(metadata))
    output.with_suffix('.tmp').replace(output)
    # Useful quality diagnostics; these describe references, not robot success.
    metadata['max_joint_step_rad'] = np.abs(np.diff(poses[:, :, 7:], axis=1)).max(axis=(1, 2)).tolist()
    metadata['root_speed_last_3s_m_s'] = np.linalg.norm(channels['body_lin_vel_w'][:, -150:, 0, :2], axis=-1).mean(-1).tolist()
    metadata['sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--planner', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seconds', type=float, default=37.)
    p.add_argument('--pillar-hold-seconds', type=float, default=2.5)
    a = p.parse_args()
    generate(a.planner, a.output, seconds=a.seconds, pillar_hold_seconds=a.pillar_hold_seconds)


if __name__ == '__main__':
    main()
