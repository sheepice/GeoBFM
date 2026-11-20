#!/usr/bin/env python3
"""Run the Revo2 scene with the saved TELEOP policy and GPU ownership guard."""
import _bootstrap  # noqa: F401
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

from da2_vibe.paths import load_settings, profile, resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/demos/revo2_pick_carry.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--duration', type=float, default=60)
    parser.add_argument('--mode', choices=('stand', 'walk', 'demo'), default='demo')
    parser.add_argument('--no-hands', action='store_true')
    parser.add_argument('--no-table', action='store_true')
    parser.add_argument('--height', type=float, default=.2)
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--arm-servo', action='store_true', help='Add a bounded Cartesian arm correction; reported separately')
    args = parser.parse_args()
    if args.mode == 'demo' and (args.no_hands or args.no_table):
        parser.error('The pickup demo requires both Revo2 hands and the table/object scene')
    if args.height < 0 or args.duration <= 0:
        parser.error('Height must be nonnegative and duration positive')
    output = Path(resolve(args.output)).resolve()
    if not args.worker:
        from vibe_grail_joint.gpu_guard import assert_gpus_idle
        locks = Path(resolve(profile().get('gpu_lock_dir', '${CACHE_ROOT}/gpu-locks')))
        locks.mkdir(parents=True, exist_ok=True)
        with (locks / f'gpu_{args.gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert_gpus_idle([args.gpu])
            output.mkdir(parents=True, exist_ok=False)
            snapshot = output / 'source'
            snapshot.mkdir()
            root = Path(__file__).resolve().parents[1]
            files = [Path(__file__), *(root/'training/src/vr_vibe').glob('revo2*.py'),
                root/'training/src/vr_vibe/native_teleop.py', root/'training/src/vr_vibe/pick_plan.py',
                root/'training/src/vr_vibe/pick_carry.py']
            hashes = {}
            for source in files:
                shutil.copy2(source, snapshot/source.name)
                hashes[str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
            (output/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                VIBE_JOINT_GUARD_PID=str(os.getpid()), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                MUJOCO_EGL_DEVICE_ID=str(args.gpu), OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1')
            with (output / 'run.log').open('w') as log:
                result = subprocess.run([sys.executable, __file__, *sys.argv[1:], '--worker'], env=env,
                    stdout=log, stderr=subprocess.STDOUT)
            print(json.dumps(dict(exit_code=result.returncode, output=str(output))), flush=True)
            return result.returncode
    from vibe_grail_joint.gpu_guard import verify_guarded_worker
    verify_guarded_worker()
    from vr_vibe.pick_carry import run
    return run(args, load_settings(args.config), output)


if __name__ == '__main__':
    raise SystemExit(main())
