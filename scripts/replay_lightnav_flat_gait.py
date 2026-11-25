"""Replay a LightNav flat-gait window or capture a full dual-camera rollout for rendering."""

import argparse
import gzip
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import _bootstrap
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from da2_vibe.paths import load_settings
from lightnav_g1_stairs import nav_camera, rgb, set_stream_velocity
from lightnav_stair_scene import build_scene, FixtureContacts
from vibe_grail_joint.da2_locomotion.evaluate import load_actor
from vibe_grail_joint.da2_locomotion.scenes import evaluation_env


def run(args):
    original = json.loads((args.reference / "result.json").read_text())
    commands = [json.loads(line) for line in (args.reference / "motion.jsonl").read_text().splitlines()]
    capture = args.capture_scene
    if capture and not original.get("wide_nav_camera"):
        raise ValueError("The unmapped figure replay requires a recorded wide navigation camera")
    seconds = commands[-1]["t"] if capture else args.seconds
    torch.set_num_threads(2)
    torch.manual_seed(original["seed"])
    cfg = load_settings(args.config)
    env, cfg, _ = evaluation_env(
        cfg,
        seed=original["seed"],
        duration=max(48.0, seconds + 1.0) if capture else seconds,
        heights=(original["tile"]["height"],),
        speeds=(0.0,),
        approaches=(original["tile"]["start"] - 1.4,),
        record_case=0,
        route="up",
        extra_sensors=(nav_camera(cfg, original.get("wide_nav_camera", False)),),
        configure_scene=build_scene,
    )
    try:
        raw = env.unwrapped
        dt = float(raw.step_dt)
        count = round(seconds / dt)
        if (
            count < 2
            or count > len(commands)
            or any(c["command_source"] != "LightNav" for c in commands[:count])
        ):
            raise ValueError("Requested interval exceeds the recorded LightNav approach")
        obs, _ = env.reset()
        actor, iteration = load_actor(cfg, obs, args.checkpoint, raw)
        motion = raw.command_manager.get_term("motion")
        stream = int(np.argmin(abs(np.asarray(cfg["planner_speeds"]) - 0.4)))
        robot = motion.robot.data
        tile = motion.cfg.random_tiles[0]
        origin = motion.tile_origins[motion.tile]
        contacts = FixtureContacts(raw.sim, tile)
        args.output.mkdir(parents=True, exist_ok=False)
        start = (robot.root_link_pos_w - origin)[0].tolist()
        if not np.allclose(start, original["start"], atol=1e-4):
            raise ValueError(f"Reset pose mismatch: {start} and {original['start']}")
        if capture:
            with TemporaryDirectory() as tmp:
                model_path = Path(tmp) / "scene.mjb"
                mujoco.mj_saveModel(raw.sim._mj_model, str(model_path))
                with gzip.open(args.output / "scene.mjb.gz", "wb") as archive:
                    archive.write(model_path.read_bytes())
            poses = [raw.sim.data.qpos[0].cpu().numpy().copy()]
            measurements = []
        with (
            imageio.get_writer(
                args.output / "side.mp4", fps=25, codec="libx264", quality=9, macro_block_size=1
            ) as writer,
            (args.output / "motion.jsonl").open("w") as log,
            torch.inference_mode(),
        ):
            for step, source in enumerate(commands[:count]):
                v, w, heading = (source[k] for k in ("v", "w", "target_heading"))
                if not math.isclose(source["t"], (step + 1) * dt, abs_tol=1e-6):
                    raise ValueError("Mismatched simulation rates")
                set_stream_velocity(motion, stream, v)
                motion.pool.navigation_heading = heading
                a, b, d, z = robot.root_link_quat_w[0].tolist()
                yaw = math.atan2(2 * (a * z + b * d), 1 - 2 * (d * d + z * z))
                error = heading - yaw
                cmd = obs["velocity_command"]
                cmd[0, :5] = cmd.new_tensor(
                    [v * math.cos(error), v * math.sin(error), w, 1 - math.cos(error), math.sin(error)]
                )
                obs, _, done, _ = env.step(actor(obs, stochastic_output=False))
                if done.any():
                    raise RuntimeError("Simulator reset during replay")
                xyz = (robot.root_link_pos_w - origin)[0].tolist()
                row = dict(
                    t=source["t"],
                    xyz=xyz,
                    gravity_z=robot.projected_gravity_b[0, 2].item(),
                    v=v,
                    w=w,
                    target_heading=heading,
                    fixture_hits=contacts.sample(),
                )
                log.write(json.dumps(row) + "\n")
                if capture:
                    feet = (motion.robot_body_pos_w[:, raw.quality.feet] - origin[:, None])[0, :, 0].tolist()
                    measurements.append([*xyz, *feet, yaw])
                    poses.append(raw.sim.data.qpos[0].cpu().numpy().copy())
                elif xyz[0] >= tile["start"] or source["xyz"][0] >= tile["start"]:
                    raise ValueError("Flat gait window extends into the stairs")
                if step % 2 == 0:
                    writer.append_data(rgb(raw, "review_camera"))
        if capture:
            np.savez_compressed(args.output / "command_replay.npz", qpos=poses,
                                origin=origin[0].cpu().numpy(), measurements=measurements)
        (args.output / "result.json").write_text(
            json.dumps(
                dict(
                    iteration=iteration,
                    checkpoint=str(args.checkpoint),
                    reference=str(args.reference),
                    start=start,
                    seconds=count * dt,
                    frames=count,
                    contacts=contacts.counts,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--seconds", type=float, default=10.0)
    mode.add_argument("--capture-scene", action="store_true",
                      help="Replay the entire command log and save the scene and poses for the unmapped figure")
    run(parser.parse_args())
