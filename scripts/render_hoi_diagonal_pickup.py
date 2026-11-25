"""Compare matched left-front, center-front and right-front G1 carry replays."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from render_hoi_pickup_offsets import DARK, TEAL, WHITE, font, scene_model

DEFAULT_CASES = ("far20_left15", "far20", "far20_right15")
LABELS = ("Left front", "Straight ahead", "Right front")
WIDTH, HEIGHT = 480, 460


def read_replays(folder, bundle, tags):
    with np.load(bundle / "motion.npz") as reference:
        nominal = reference["box_pose"][0, :2]
        forward = nominal - reference["qpos"][0, :2]
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0]])
    runs, identities, offsets = [], [], []
    for tag in tags:
        result = json.loads((folder / tag / "evaluation.json").read_text())
        with np.load(folder / tag / "closed_loop.npz") as recording:
            assert recording["fps"] == 50 and result["vision"] == "normal"
            poses = recording["qpos"].copy()
        identities.append(
            (
                result["checkpoint_sha256"],
                result["seed"],
                result["selected"]["env"],
                result["initial_robot_qpos_sha256"],
            )
        )
        advance, side = result["box_offset_forward_left_m"]
        offsets.append((advance, side))
        np.testing.assert_allclose(poses[0, -7:-5], nominal + forward * advance + left * side, atol=1e-5)
        if runs:
            np.testing.assert_allclose(poses[0, :-7], runs[0][0, :-7], atol=1e-7)
        runs.append(poses)
    assert len(set(identities)) == 1
    advance, sides = np.array(offsets).T
    np.testing.assert_allclose(advance, advance[0])
    assert sides[0] > 0 and sides[1] == 0 and np.isclose(sides[2], -sides[0])
    return runs, nominal, forward, left, advance[0], sides


def floor_line(scene, a, b):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).ravel(),
        np.array([0.42, 0.90, 0.84, 1.0]),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.005, np.r_[a, 0.008], np.r_[b, 0.008])
    scene.ngeom += 1


def render(bundle, trials, output, tags=DEFAULT_CASES):
    runs, nominal, forward, left, advance, sides = read_replays(trials, bundle, tags)
    model = scene_model(bundle)
    model.vis.global_.offheight = HEIGHT
    data = mujoco.MjData(model)
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    options.sitegroup[:] = 0
    cam = mujoco.MjvCamera()
    fps = 25
    assert all(model.nq == len(poses[0]) for poses in runs)
    frames = max(map(len, runs))
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:

        def compose(frame, overhead):
            canvas = Image.new("RGB", (WIDTH * 3, HEIGHT + 108), DARK)
            draw = ImageDraw.Draw(canvas)
            draw.text((20, 12), "G1  /  Three pickup positions", fill=WHITE, font=font(29))
            detail = (
                "Start positions   |   Overhead" if overhead else f"Fixed camera   |   {frame / 50:.2f} s"
            )
            draw.text((WIDTH * 3 - 20, 18), detail, anchor="ra", fill=TEAL, font=font(21))
            cam.azimuth = 180 if overhead else 150
            cam.elevation = -88 if overhead else -29
            cam.distance = 2.55 if overhead else 3.42
            cam.lookat[:] = [-0.21, -0.065, 0.18] if overhead else [0.22, -0.48, 0.61]
            for i, (name, offset) in enumerate(zip(LABELS, sides, strict=True)):
                data.qpos[:] = runs[i][min(frame, len(runs[i]) - 1)]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam, scene_option=options)
                for start in np.arange(-0.46, 0.63, 0.10):
                    floor_line(renderer.scene, nominal + forward * start, nominal + forward * (start + 0.055))
                if overhead:
                    # Dimension ticks on the far side of the box remain visible from above.
                    origin = nominal + (advance + 0.26) * forward
                    floor_line(
                        renderer.scene, origin - (sides[0] + 0.06) * left, origin + (sides[0] + 0.06) * left
                    )
                    for tick in sides:
                        p = origin + tick * left
                        floor_line(renderer.scene, p - 0.018 * forward, p + 0.018 * forward)
                canvas.paste(Image.fromarray(renderer.render()), (i * WIDTH, 54))
                draw.text((i * WIDTH + 14, HEIGHT + 68), name, font=font(22), fill=WHITE)
                if frame >= len(runs[i]):
                    draw.text(
                        (i * WIDTH + 12, 64),
                        "Episode ended",
                        font=font(20),
                        fill=WHITE,
                        stroke_width=2,
                        stroke_fill=DARK,
                    )
                desc = f"+{advance * 100:.0f} cm forward   /   {abs(offset) * 100:.0f} cm " + (
                    "left" if offset > 0 else "right" if offset < 0 else "lateral"
                )
                draw.text(((i + 1) * WIDTH - 12, HEIGHT + 72), desc, anchor="ra", font=font(16), fill=TEAL)
                if i:
                    draw.line([(i * WIDTH, 54), (i * WIDTH, HEIGHT + 54)], fill=DARK, width=3)
            return canvas

        opening = compose(0, True)
        output.parent.mkdir(exist_ok=True, parents=True)
        opening.save(output.with_suffix(".jpg"), quality=92)
        with imageio.get_writer(
            output,
            fps=fps,
            codec="libx264",
            macro_block_size=None,
            ffmpeg_params=["-threads", "3", "-crf", "18", "-movflags", "+faststart"],
        ) as writer:
            for _ in range(fps * 2):
                writer.append_data(np.asarray(opening))
            for frame in range(0, frames, 2):
                writer.append_data(np.asarray(compose(frame, False)))
            final = np.asarray(compose(frames - 1, False))
            for _ in range(12):
                writer.append_data(final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("bundle", "trials", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--cases", nargs=3, default=DEFAULT_CASES, metavar=("LEFT", "CENTER", "RIGHT"))
    args = parser.parse_args()
    render(args.bundle, args.trials, args.output, args.cases)
