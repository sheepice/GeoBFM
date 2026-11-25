"""Record a language-conditioned LightNav-0 + DA2 G1 stair rollout in MuJoCo.

Run the upstream LightNav-0 VLN WebSocket server separately and pass its address
with --server. The default shared-RGB mode sends the same head camera frame to
the planner and the locomotion policy.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import base64
import hashlib
import io
import json
import math
from pathlib import Path

import _bootstrap
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from websockets.sync.client import connect

from lightnav_stair_scene import build_scene, fixtures, FixtureContacts, AisleSpeedGuide, PhysicsStepContacts
from da2_vibe.paths import load_settings
from vibe_grail_joint.da2_locomotion.scenes import evaluation_env
from vibe_grail_joint.da2_locomotion.evaluate import load_actor


def nav_camera(settings, wide=False):
    from mjlab.sensor import CameraSensorCfg
    from scipy.spatial.transform import Rotation

    pitch = math.radians(0 if wide else 15)
    rotation = np.column_stack(
        ((0, -1, 0), (math.sin(pitch), 0, math.cos(pitch)), (-math.cos(pitch), 0, math.sin(pitch)))
    )
    return CameraSensorCfg(
        name="nav_cam",
        parent_body=settings["camera"]["parent_body"],
        pos=(0.14, 0.01, 0.44),
        quat=tuple(Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]),
        width=448,
        height=256,
        fovy=89.42 if wide else 60.0,
        data_types=("rgb",),
        enabled_geom_groups=(0, 2),
    )


def request(socket, action, payload):
    socket.send(json.dumps(dict(action=action, data=payload)))
    reply = json.loads(socket.recv(timeout=240))
    if reply["action"] != action or reply["data"].get("rc") != 0:
        raise RuntimeError(f"LightNav {action} failed: {reply}")
    return reply["data"]


def predict(socket, rgb, instruction, seq, lossless=False):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG" if lossless else "JPEG", quality=90)
    if lossless:
        with Image.open(io.BytesIO(buf.getvalue())) as decoded:
            if not np.array_equal(np.asarray(decoded.convert("RGB")), rgb):
                raise RuntimeError("Navigation payload differs from the DA2 frame")
    data = request(
        socket,
        "next",
        dict(seq=seq, instruction=instruction, image=base64.b64encode(buf.getvalue()).decode("ascii")),
    )
    if data["seq"] != seq or not isinstance(data["stop"], bool):
        raise ValueError("Invalid LightNav response")
    path = np.asarray(data["actions"]["actions"], dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or not len(path) or not np.isfinite(path).all():
        raise ValueError("Invalid LightNav trajectory")
    return data, path


def command(path, stop, max_speed=0.4):
    if stop:
        return 0.0, 0.0
    # The API returns cumulative ego poses. We use 0.1 s/row as a tracking convention.
    # Track the third waypoint to smooth the quantized first displacement.
    count = min(3, len(path))
    x, y, yaw = path[count - 1]
    duration = count * 0.1
    v = float(np.clip(x / duration, 0, max_speed))
    w = float(np.clip((yaw + math.atan2(y, max(0.05, x))) / (2 * duration), -0.35, 0.35))
    return v, w


def set_stream_velocity(c, stream, speed):
    # The tokenizer clamps lookahead frames to _clip_ids even in online mode.
    c.stream.fill_(stream)
    c._clip_ids.fill_(c.replay_clips + stream)
    c.pool.speeds[stream] = speed
    c.speed.fill_(speed)
    c.fixed_speeds.fill_(speed)


class SharedHeadRGB:
    """Latch the exact uint8 input most recently consumed by DA2, including reset."""

    def __init__(self, raw):
        self.raw = raw
        self.frame = None
        self.step = None
        self.handle = raw.da2_visual.encoder.register_forward_pre_hook(self.capture)

    def capture(self, module, inputs):
        frame = inputs[0]
        if frame.dtype != torch.uint8 or tuple(frame.shape) != (1, 112, 112, 3):
            raise ValueError("Unexpected DA2 camera input")
        self.frame = frame[0].detach().cpu().numpy().copy()
        self.step = int(self.raw.common_step_counter)

    def latest(self):
        if self.frame is None or self.step != int(self.raw.da2_visual.history.last_step[0]):
            raise RuntimeError("Navigation frame missing or out of sync with DA2")
        return self.frame


def rgb(raw, name):
    return raw.scene[name].data.rgb[0].detach().cpu().numpy().copy()


def projected_path(raw, camera, ground, xy, color):
    image = Image.fromarray(rgb(raw, camera))
    idx = raw.scene[camera].camera_idx
    pos = raw.sim.data.cam_xpos[0, idx].cpu().numpy()
    rot = raw.sim.data.cam_xmat[0, idx].cpu().numpy().reshape(3, 3)
    coords = (np.column_stack((xy, ground)) - pos) @ rot
    depth = -coords[:, 2]
    f = 0.5 * image.height / math.tan(math.radians(float(raw.sim._mj_model.cam_fovy[idx])) / 2)
    visible = depth > 0.03
    points = np.column_stack(
        (
            image.width / 2 + f * coords[:, 0] / np.maximum(depth, 0.03),
            image.height / 2 - f * coords[:, 1] / np.maximum(depth, 0.03),
        )
    )
    draw = ImageDraw.Draw(image)
    for i in range(1, len(xy)):
        if visible[i - 1] and visible[i]:
            vertices = [tuple(points[i - 1]), tuple(points[i])]
            draw.line(vertices, fill="#183236", width=6)
            draw.line(vertices, fill=color, width=3)
    return image


def film_frame(raw, instruction, t, v, w, xyz, tile, reason, track, plan, origin, plan_label, title, camera):
    plan_color = "#ffca54"
    goal = tile["goal"]
    view = Image.new("RGB", (960, 720), (22, 31, 32))
    view.paste(Image.fromarray(rgb(raw, "review_camera")).resize((640, 360)), (0, 82))
    # Terrain heights and odometry are used only to draw predicted points in the video.
    edges = np.asarray(tile["edges"])
    elevations = np.r_[tile["base"], tile["base"] + np.cumsum(tile["jumps"])]
    z = elevations[np.searchsorted(edges, plan[:, 0], side="right")] + origin[2] + 0.035
    camera_image = projected_path(raw, camera, z, plan + origin[:2], plan_color)
    size = (174, 174) if camera == "head_cam" else (304, 174)
    view.paste(camera_image.resize(size), (648, 82))
    view.paste(Image.fromarray(rgb(raw, "head_cam")).resize((162, 162)), (648, 288))
    draw = ImageDraw.Draw(view)
    for point, text, size in [
        ((18, 9), title, 25),
        ((18, 48), instruction, 15),
        ((649, 259), "Navigation RGB + " + plan_label, 15),
        ((817, 354), "Shared RGB" if camera == "head_cam" else "DA2 RGB", 16),
        ((18, 452), f"Simulation {t:5.2f} s     Command {v:.2f} m/s    {w:+.2f} rad/s", 20),
        (
            (18, 489),
            f"Pelvis x {xyz[0]:.2f} m    y {xyz[1]:+.2f} m    z {xyz[2]:.2f} m     Goal x {goal:.2f} m",
            18,
        ),
        ((18, 533), reason, 18),
        ((18, 575), "Top view", 17),
    ]:
        draw.text(point, text, fill="white", font_size=size)
    scale = 860 / (goal + 1.35)

    def pixel(p):
        return (55 + scale * p[0], 654 - 29 * p[1])

    for item in fixtures(tile):
        bounds = [
            pixel((item.x - item.length / 2, item.y + item.width / 2)),
            pixel((item.x + item.length / 2, item.y - item.width / 2)),
        ]
        draw.rectangle(bounds, fill="#bf904e" if item.style == "crate" else "#e67338", outline="#ffdec0")
    for x in edges:
        u = pixel((x, 0))[0]
        draw.line((u, 605, u, 701), fill="#816b54", width=2)
    u, y = pixel((goal, 0))
    draw.ellipse((u - 6, y - 6, u + 6, y + 6), outline="white", width=2)
    draw.text((u - 17, 575), "Goal", fill="white", font_size=15)
    draw.line([pixel(p) for p in track], fill="#56dddf", width=4)
    draw.line([pixel(p) for p in plan], fill=plan_color, width=3)
    for p in plan[1:]:
        u, y = pixel(p)
        draw.ellipse((u - 2, y - 2, u + 2, y + 2), fill=plan_color)
    draw.text((118, 575), "G1 actual", fill="#56dddf", font_size=17)
    draw.text((233, 575), plan_label, fill=plan_color, font_size=17)
    return np.asarray(view)


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    cfg = load_settings(args.config)
    shared = not (args.wide_nav_camera or args.separate_nav_camera)
    camera = "head_cam" if shared else "nav_cam"

    def configure_demo_scene(env_cfg, tile):
        build_scene(env_cfg, tile)
        if args.mount_pitch_up_deg:
            from scipy.spatial.transform import Rotation

            sensors = []
            for sensor in env_cfg.scene.sensors:
                if sensor.name == "head_cam":
                    mount = Rotation.from_quat(np.array(sensor.quat)[[1, 2, 3, 0]])
                    mount = mount * Rotation.from_euler("x", args.mount_pitch_up_deg, degrees=True)
                    sensor = replace(sensor, quat=tuple(mount.as_quat()[[3, 0, 1, 2]]))
                sensors.append(sensor)
            env_cfg.scene.sensors = tuple(sensors)

    args.output.mkdir(parents=True, exist_ok=False)
    env, cfg, _ = evaluation_env(
        cfg,
        seed=args.seed,
        duration=args.duration,
        heights=(args.height,),
        speeds=(0.0,),
        approaches=(args.approach,),
        record_case=0,
        route="up",
        extra_sensors=() if shared else (nav_camera(cfg, args.wide_nav_camera),),
        configure_scene=configure_demo_scene,
    )
    raw = env.unwrapped
    shared_frames = SharedHeadRGB(raw) if shared else None
    try:
        obs, _ = env.reset()
        actor, iteration = load_actor(cfg, obs, args.checkpoint, raw)
        c = raw.command_manager.get_term("motion")
        tile = c.cfg.random_tiles[0]
        goal = tile["goal"]
        for name in dict.fromkeys(("head_cam", camera, "review_camera")):
            Image.fromarray(rgb(raw, name)).save(args.output / (name + ".png"))
        contacts = FixtureContacts(raw.sim, tile)
        if args.preview:
            return
        guide = AisleSpeedGuide(tile, args.clearance_m) if args.aisle_clearance else None
        physics = PhysicsStepContacts(raw, tile) if guide or shared else None
        speeds = np.asarray(cfg["planner_speeds"])
        dt = float(raw.step_dt)
        robot = c.robot.data
        start = (robot.root_link_pos_w - c.tile_origins[c.tile])[0].tolist()
        track = [start[:2]]
        origin = c.tile_origins[c.tile][0].cpu().numpy().copy()
        seq = 0
        stop = False
        held = 0
        result = "timeout"
        stop_ticks = 0
        heading_ref = None
        fixed = args.constant_speed is not None
        source = "constant_velocity" if fixed else "LightNav"
        plan_label = "Command direction" if fixed else "LightNav prediction"
        title = f"GeoBFM G1  /  DA2 {iteration}  /  " + ("Constant velocity" if fixed else "LightNav-0")
        if guide:
            title += " + mapped obstacle clearance"
        instruction = (
            f"Constant forward speed {args.constant_speed:.2f} m/s; zero turn rate."
            if fixed
            else args.instruction
        )
        transport = (
            nullcontext(None)
            if fixed
            else connect(args.server, open_timeout=20, close_timeout=3, proxy=None, max_size=2**22)
        )
        stream = int(np.argmin(abs(speeds - 0.4)))
        with (
            transport as socket,
            (args.output / "predictions.jsonl").open("w", buffering=1) as predictions,
            (args.output / "motion.jsonl").open("w", buffering=1) as motion,
            imageio.get_writer(
                args.output / "navigation.mp4", fps=25, codec="libx264", quality=8, macro_block_size=1
            ) as writer,
            torch.inference_mode(),
        ):
            if not fixed:
                request(socket, "login", dict(clientId="geobfm-g1-mujoco"))
                request(socket, "reset", {})
            for step in range(round(args.duration / dt)):
                if step % args.replan_steps == 0 and not stop and not fixed:
                    frame = shared_frames.latest() if shared else rgb(raw, camera)
                    data, path = predict(socket, frame, args.instruction, seq, lossless=shared)

                    seq += 1
                    raw_v, raw_w = command(path, data["stop"], args.max_speed)
                    v, w = raw_v, raw_w
                    stop = data["stop"]
                    a, b, d, z = robot.root_link_quat_w[0].tolist()
                    yaw = math.atan2(2 * (a * z + b * d), 1 - 2 * (d * d + z * z))
                    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
                    capture = robot.root_link_pos_w[0, :2].cpu().numpy() - origin[:2]
                    plan = np.vstack((capture, capture + path[:, :2] @ rotation.T))
                    slip = 0.0
                    if len(track) > 45:
                        displacement = capture - np.asarray(track[-46])
                        if np.linalg.norm(displacement) > 0.13:
                            course = math.atan2(displacement[1], displacement[0])
                            slip = float(np.clip(math.remainder(course - yaw, 2 * math.pi), -0.35, 0.35))
                    record = dict(
                        camera=camera,
                        rgb_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                        da2_capture_step=shared_frames.step if shared else None,
                        slip_rad=slip,
                        step=step,
                        sim_time=step * dt,
                        response=data,
                        v=v,
                        w=w,
                        capture_xy=capture.tolist(),
                        capture_yaw=yaw,
                        plan_xy=plan.tolist(),
                    )
                    predictions.write(json.dumps(record) + "\n")
                    print(f"{step * dt:.2f}s  v={v:.2f}  w={w:+.2f}  stop={stop}", flush=True)
                if fixed:
                    raw_v, raw_w = args.constant_speed, 0.0
                v, w = raw_v, raw_w
                yawq = robot.root_link_quat_w[0].tolist()
                a, b, d, z = yawq
                yaw = math.atan2(2 * (a * z + b * d), 1 - 2 * (d * d + z * z))
                if heading_ref is None:
                    heading_ref = yaw
                heading_ref += w * dt
                if fixed or args.heading_tracker == "integrated":
                    target_heading = heading_ref
                else:
                    target_heading = record["capture_yaw"] + w * 0.65 - record["slip_rad"]
                if guide and not stop:
                    v, w, target_heading = guide.update(track[-1], yaw, raw_v, raw_w)
                set_stream_velocity(c, stream, v)
                if fixed:
                    plan = np.array(
                        [
                            track[-1],
                            np.asarray(track[-1])
                            + 0.9 * np.array([math.cos(target_heading), math.sin(target_heading)]),
                        ]
                    )
                heading_error = target_heading - yaw
                c.pool.navigation_heading = target_heading
                cmd = obs["velocity_command"]
                cmd[0, :5] = cmd.new_tensor(
                    [
                        v * math.cos(heading_error),
                        v * math.sin(heading_error),
                        w,
                        1 - math.cos(heading_error),
                        math.sin(heading_error),
                    ]
                )
                if physics:
                    physics.step_hits.clear()
                action = actor(obs, stochastic_output=False)
                obs, _, done, _ = env.step(action)
                if done.any():
                    raise RuntimeError("Unexpected simulator reset")
                xyz = (robot.root_link_pos_w - c.tile_origins[c.tile])[0].tolist()
                feet = (c.robot_body_pos_w[:, raw.quality.feet] - c.tile_origins[c.tile, None, :])[0]
                supported = bool(raw.quality.sensor_support.any())
                gravity = float(robot.projected_gravity_b[0, 2])
                hits = contacts.sample()
                near = math.hypot(xyz[0] - goal, xyz[1]) <= 0.60
                on_landing = bool((feet[:, 0] >= tile["edges"][-1]).all())
                held = held + 1 if near and on_landing and gravity < -0.85 and supported else 0
                stop_ticks = stop_ticks + 1 if stop else 0
                t = (step + 1) * dt
                track.append(xyz[:2])
                motion.write(
                    json.dumps(
                        dict(
                            t=t,
                            xyz=xyz,
                            v=v,
                            w=w,
                            yaw=yaw,
                            target_heading=target_heading,
                            stop=stop,
                            command_source=source,
                            requested_v=raw_v,
                            requested_w=raw_w,
                            fixture_hits_substep=sorted(physics.step_hits) if physics else None,
                            fixture_hits=hits,
                            feet_x=feet[:, 0].tolist(),
                            supported=supported,
                            gravity_z=gravity,
                            goal_hold=held,
                        )
                    )
                    + "\n"
                )
                if held * dt >= 0.6:
                    result = "arrived"
                elif gravity > -0.4 or xyz[2] < 0.25:
                    result = "fell"
                elif stop_ticks * dt >= 1.5:
                    result = "stopped_before_goal"
                if step % 2 == 0 or result != "timeout":
                    label = {
                        "arrived": "Reached upper landing",
                        "fell": "Fall",
                        "stopped_before_goal": "Model stopped before goal",
                    }.get(result, "VLN stop" if stop else "Following predicted path")
                    if fixed and result == "timeout":
                        label = "Tracking constant velocity"
                    frame = film_frame(
                        raw,
                        instruction,
                        t,
                        v,
                        w,
                        xyz,
                        tile,
                        label,
                        track,
                        plan,
                        origin,
                        plan_label,
                        title,
                        camera,
                    )
                    writer.append_data(frame)
                if result != "timeout":
                    break
        summary = dict(
            outcome=result,
            sim_seconds=t,
            queries=seq,
            start=start,
            end=xyz,
            goal_xy=[goal, 0],
            distance_to_goal_m=math.hypot(xyz[0] - goal, xyz[1]),
            model_stop=stop,
            held_seconds=held * dt,
            tile=tile,
            seed=args.seed,
            checkpoint=str(args.checkpoint),
            iteration=iteration,
            sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            instruction=instruction,
            command_source=source,
            constant_speed=args.constant_speed,
            max_nav_speed=args.max_speed,
            replan_steps=args.replan_steps,
            heading_tracker=args.heading_tracker,
            shared_rgb=shared,
            onboard_navigation_camera=camera,
            nav_wire_codec="PNG" if shared else "JPEG",
            nominal_camera_contract=cfg["camera"],
            mount_pitch_up_deg=args.mount_pitch_up_deg,
            applied_head_quat_wxyz=raw.sim._mj_model.cam_quat[raw.scene["head_cam"].camera_idx].tolist(),
            applied_head_fovy=float(raw.sim._mj_model.cam_fovy[raw.scene["head_cam"].camera_idx]),
            wide_nav_camera=args.wide_nav_camera,
            mapped_guide=bool(guide),
            fixtures=[asdict(item) for item in fixtures(tile)],
            fixture_contact_frames_50hz=contacts.counts,
        )
        if physics:
            summary.update(
                clearance_guide_m=args.clearance_m if guide else None,
                physics_dt_s=float(raw.physics_dt),
                physics_samples=physics.samples,
                fixture_contact_physics_steps=physics.observer.counts,
                min_fixture_surface_distance_m=physics.min_distance,
                closest_geom_pair=physics.closest,
            )
            if physics.samples != round(t / raw.physics_dt):
                raise RuntimeError("Incomplete physics-step contact log")
        (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(result, xyz, flush=True)
    finally:
        if shared_frames:
            shared_frames.handle.remove()
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--server", default="ws://127.0.0.1:18050")
    p.add_argument(
        "--instruction",
        default="Walk between the wooden crates and the orange barrier, then climb the stairs to the upper landing.",
    )
    p.add_argument(
        "--constant-speed",
        type=float,
        help="Apply a fixed forward velocity for the entire recording, without a navigation server",
    )
    p.add_argument("--heading-tracker", choices=("relative", "integrated"), default="relative")
    camera_args = p.add_mutually_exclusive_group()
    camera_args.add_argument(
        "--shared-rgb", action="store_true", help="Use the same head RGB pixels as DA2 (default)"
    )
    camera_args.add_argument(
        "--separate-nav-camera", action="store_true", help="Reproduce the separate narrow navigation camera"
    )
    camera_args.add_argument(
        "--wide-nav-camera",
        action="store_true",
        help="Use a level navigation camera with approximately 120 deg horizontal FOV",
    )
    p.add_argument(
        "--max-speed", type=float, default=0.4, help="Cap on the model-requested forward velocity in m/s"
    )
    p.add_argument(
        "--aisle-clearance",
        action="store_true",
        help="Use mapped obstacle positions for lateral clearance and 0.25/0.4 m/s speed caps",
    )
    p.add_argument(
        "--clearance-m", type=float, default=0.58, help="Pelvis path clearance from mapped fixture edges"
    )
    p.add_argument(
        "--mount-pitch-up-deg",
        type=float,
        default=0.0,
        help="Explore a physically tilted single head camera in simulation",
    )
    p.add_argument("--height", type=float, default=0.15)
    p.add_argument("--approach", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=431113)
    p.add_argument("--duration", type=float, default=48.0)
    p.add_argument("--replan-steps", type=int, default=12)
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()
    if args.replan_steps <= 0 or args.duration <= 0:
        p.error("Replanning interval and duration must be positive")
    if not 0 < args.max_speed <= 0.7:
        p.error("Navigation speed limit must be in (0, 0.7] m/s")
    if args.constant_speed is not None and not 0 <= args.constant_speed <= 0.7:
        p.error("Constant speed must be in [0, 0.7] m/s")
    if args.mount_pitch_up_deg and (args.wide_nav_camera or args.separate_nav_camera):
        p.error("Mount pitch comparison requires shared RGB")
    run(args)
