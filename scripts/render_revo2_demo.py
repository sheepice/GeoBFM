#!/usr/bin/env python3
"""Render actual recorded qpos at 1x speed with an independent hand close-up."""
import _bootstrap  # noqa: F401
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

from da2_vibe.paths import profile, resolve


def render(args):
    import imageio.v2 as iio
    import matplotlib
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    run = Path(resolve(args.run)).resolve()
    output = Path(resolve(args.output)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace = json.loads((run / 'trace.json').read_text())
    poses = np.load(run / 'trajectory.npz')['qpos']
    times = np.array([row['t'] for row in trace])
    if len(poses) != len(trace) or not np.allclose(np.diff(times), .02):
        raise ValueError('Expected matching trace/qpos at 50 Hz')
    end = min(args.end, times[-1])
    if not 0 <= args.start < end:
        raise ValueError('Invalid export interval')
    indices = np.flatnonzero((times >= args.start) & (times <= end))[::2]
    if not args.preview and (output / 'revo2_pick_carry_stairs.mp4').exists():
        raise FileExistsError('Use a new output directory to preserve the previous export')
    model = mujoco.MjModel.from_binary_path(str(run / 'scene.mjb'))
    data = mujoco.MjData(model)
    overview = mujoco.Renderer(model, height=720, width=1280)
    detail = mujoco.Renderer(model, height=300, width=400)
    options = mujoco.MjvOption()
    options.geomgroup[3] = False
    wide, close = mujoco.MjvCamera(), mujoco.MjvCamera()
    wide.azimuth, wide.elevation = 120, -20
    close.azimuth, close.elevation, close.distance = 115, -18, .58
    font_path = Path(matplotlib.__file__).parent / 'mpl-data/fonts/ttf/DejaVuSans.ttf'
    title = ImageFont.truetype(str(font_path), 25)
    small = ImageFont.truetype(str(font_path), 19)
    payload = model.body('payload').id
    wrist = model.body('right_wrist_yaw_link').id
    phases = dict(SETTLE='Stand', RAISE_HAND='Raise hand', PREGRASP='Reach above bottle',
        LOWER_HAND='Lower hand', APPROACH='Approach bottle', CLOSE='Close fingers',
        LIFT='Lift bottle', CARRY_UPSTAIRS='Carry upstairs', COMPLETE='Hold on landing')

    def frame_at(i):
        data.qpos[:] = poses[i]
        mujoco.mj_forward(model, data)
        row, root = trace[i], poses[i, :3]
        widening = float(np.clip((times[i] - 24) / 5, 0, 1))
        widening = widening*widening*(3-2*widening)
        wide.distance = 3.2 + .8*widening
        wide.lookat[:] = [root[0]+.35, -.12, root[2]+.05]
        overview.update_scene(data, camera=wide, scene_option=options)
        frame = Image.fromarray(overview.render())
        # The inset shows the same qpos sample; only its camera differs.
        center = .5*(data.xpos[payload]+data.xpos[wrist])
        close.lookat[:] = center
        detail.update_scene(data, camera=close, scene_option=options)
        inset = Image.fromarray(detail.render())
        frame.paste(inset, (856, 355))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, 1280, 53), fill=(16, 23, 34))
        draw.text((22, 12), 'G1 + BrainCo Revo2 | Pick up, carry, climb', font=title, fill='white')
        draw.text((1008, 15), f't = {times[i]:05.2f} s  |  1x', font=small, fill='#b8d6e6')
        draw.rounded_rectangle((20, 71, 415, 145), radius=9, fill=(16, 23, 34))
        draw.text((35, 81), phases.get(row['phase'], row['phase']), font=title, fill='white')
        draw.text((35, 116), f"Forward command: {row['speed_command']:.1f} m/s", font=small, fill='#b8d6e6')
        draw.rectangle((854, 322, 1258, 355), fill=(16, 23, 34))
        draw.text((866, 328), 'Right hand | same recorded state', font=small, fill='white')
        draw.rectangle((854, 354, 1258, 657), outline='#b8d6e6', width=2)
        draw.rectangle((0, 677, 1280, 720), fill=(16, 23, 34))
        draw.text((22, 688), 'Body: model_4400 | Hand targets + fingers: scripted | 60 g bottle | 6 x 20 cm stairs',
            font=small, fill='white')
        return frame

    writer = None
    try:
        if args.preview:
            for timestamp in args.preview:
                i = int(np.argmin(np.abs(times-timestamp)))
                frame_at(i).save(output / f'preview_{times[i]:05.2f}.jpg')
            return
        writer = iio.get_writer(output / 'revo2_pick_carry_stairs.mp4', fps=25,
            codec='libx264', quality=8, macro_block_size=1, ffmpeg_params=['-movflags', '+faststart'])
        for count, i in enumerate(indices):
            frame = frame_at(i)
            writer.append_data(np.asarray(frame))
            if count % 125 == 0:
                print(f'Rendered {count}/{len(indices)} frames (t={times[i]:.2f}s)', flush=True)
            if count in (0, len(indices)-1) or i % 250 == 0:
                frame.save(output / f'frame_{times[i]:05.2f}.jpg')
    finally:
        if writer is not None:
            writer.close()
        overview.close()
        detail.close()
    source = Path(__file__).resolve()
    shutil.copy2(source, output / source.name)
    metadata = dict(source_run=str(run), source_scene_sha256=hashlib.sha256((run/'scene.mjb').read_bytes()).hexdigest(),
        source_trajectory_sha256=hashlib.sha256((run/'trajectory.npz').read_bytes()).hexdigest(),
        render_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        first_source_time_s=float(times[indices[0]]), last_source_time_s=float(times[indices[-1]]),
        fps=25, frames=len(indices), video_duration_s=len(indices)/25, playback_speed=1,
        source_stride=2, source_control_hz=50, physics_rerun=False, poses_modified=False,
        inset='Independent camera, same qpos sample as overview', gpu=args.gpu)
    (output / 'render_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--end', type=float, default=40)
    parser.add_argument('--preview', type=float, nargs='+')
    args = parser.parse_args()
    from vibe_grail_joint.gpu_guard import assert_gpus_idle
    locks = Path(resolve(profile().get('gpu_lock_dir', '${CACHE_ROOT}/gpu-locks')))
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f'gpu_{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_gpus_idle([args.gpu])
        os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
            MUJOCO_EGL_DEVICE_ID=str(args.gpu), CUDA_VISIBLE_DEVICES=str(args.gpu),
            CUDA_DEVICE_ORDER='PCI_BUS_ID')
        render(args)


if __name__ == '__main__':
    main()
