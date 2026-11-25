"""Replay the four recorded box starts at true scale and with a fixed world camera."""

import argparse
from functools import lru_cache
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 768, 400
HEADER = 64
DARK = "#152a2d"
WHITE = "#f3f6ef"
TEAL = "#79e1d3"
TAGS = ("nominal", "far10", "far20", "far30")


@lru_cache
def font(size):
    return ImageFont.load_default(size=size)


def load_trials(folder):
    trajectories = []
    for i, tag in enumerate(TAGS):
        entry = json.loads((folder / tag / "evaluation.json").read_text())
        if i == 0:
            reference = entry
        for k in ("iteration", "checkpoint_sha256", "initial_robot_qpos_sha256", "seed"):
            assert entry[k] == reference[k]
        assert entry["selected"]["env"] == reference["selected"]["env"]
        assert entry["selected"]["placed"] and entry["vision"] == "normal"
        with np.load(folder / tag / "closed_loop.npz") as f:
            assert int(f["fps"]) == 50
            trajectories.append(f["qpos"].copy())
    trials = np.stack(trajectories)
    np.testing.assert_allclose(trials[:, 0, :-7], np.repeat(trials[:1, 0, :-7], 4, axis=0), atol=1e-7)
    dist = np.linalg.norm(trials[:, 0, -7:-5] - trials[0, 0, -7:-5], axis=1)
    np.testing.assert_allclose(dist, np.arange(4) / 10, atol=1e-5)
    return trials, reference["iteration"]


def scene_model(bundle):
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    spec.attach(
        mujoco.MjSpec.from_file(str(bundle / "box.xml")), prefix="box/", frame=spec.worldbody.add_frame()
    )
    spec.add_texture(
        name="tiles",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=128,
        height=128,
        rgb1=[0.33, 0.39, 0.41],
        rgb2=[0.36, 0.42, 0.44],
    )
    mat = spec.add_material(name="tiles", texrepeat=[2, 2], texuniform=True)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tiles"
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1], material="tiles"
    )
    spec.worldbody.add_light(
        pos=[0, 1.5, 4], dir=[0, -0.2, -1], ambient=[0.4] * 3, diffuse=[0.75] * 3, castshadow=True
    )
    spec.visual.global_.offwidth = WIDTH
    spec.visual.global_.offheight = HEIGHT
    model = spec.compile()
    model.vis.headlight.ambient[:] = 0.45
    return model


def ground_project(scene, points):
    left, right = scene.camera
    eye = (left.pos + right.pos) * 0.5
    forward = left.forward / np.linalg.norm(left.forward)
    horizontal = np.cross(forward, left.up)
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(horizontal, forward)
    v = points - eye
    focal = HEIGHT * left.frustum_near / (left.frustum_top - left.frustum_bottom)
    return (
        np.column_stack(
            [
                WIDTH / 2 + focal * (v @ horizontal) / (v @ forward),
                HEIGHT / 2 - focal * (v @ vertical) / (v @ forward),
            ]
        )
        .round()
        .astype(int)
    )


def render(args):
    traces, iteration = load_trials(args.trials)
    model = scene_model(args.bundle)
    data = mujoco.MjData(model)
    assert model.nq == traces.shape[-1] and model.joint("box/object_root").qposadr.item() == 36
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = -90, -13, 2.85
    cam.lookat[:] = [0.18, -0.37, 0.67]
    opt = mujoco.MjvOption()
    opt.geomgroup[3:] = 0
    opt.sitegroup[:] = 0
    floor = traces[:, 0, -7:-4].copy()
    floor[:, 2] = 0
    ruler = floor.copy()
    ruler[:, 1] += 0.37
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:

        def panel(k, t, detail=False):
            data.qpos[:] = traces[k, t]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam, scene_option=opt)
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            ticks = ground_project(renderer.scene, ruler)
            centers = ground_project(renderer.scene, floor)
            x, y = map(int, ticks[k])
            draw.line([tuple(ticks[0] - [8, 0]), tuple(ticks[-1] + [8, 0])], fill=WHITE, width=2)
            for tx, ty in ticks:
                draw.line([(tx, ty - 3), (tx, ty + 4)], fill=WHITE, width=2)
            draw.line([tuple(centers[k]), (x, y)], fill=TEAL, width=2)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=TEAL)
            if detail:
                # Render labels on the enlarged frame; tick coordinates stay tied to world space.
                img = img.resize((2 * WIDTH, 2 * HEIGHT), Image.Resampling.LANCZOS)
                draw = ImageDraw.Draw(img)
                for n, (tx, ty) in enumerate(ticks):
                    draw.text(
                        (2 * tx, 2 * ty + 10),
                        str(n * 10),
                        fill=WHITE,
                        stroke_width=2,
                        stroke_fill=DARK,
                        font=font(20),
                        anchor="mt",
                    )
                draw.text(
                    (2 * ticks[-1, 0] + 30, 2 * ticks[-1, 1] + 10),
                    "cm",
                    font=font(19),
                    fill=WHITE,
                    stroke_width=2,
                    stroke_fill=DARK,
                )
            else:
                draw.rounded_rectangle([14, 12, 137, 48], 5, fill=DARK)
                draw.text((24, 16), f"{k * 10} cm", fill=WHITE, font=font(25))
            return img

        def canvas(title, right):
            im = Image.new("RGB", (WIDTH * 2, HEIGHT * 2 + HEADER), DARK)
            draw = ImageDraw.Draw(im)
            draw.text((24, 13), title, font=font(29), fill=WHITE)
            draw.text((WIDTH * 2 - 24, 22), right, font=font(20), fill=TEAL, anchor="ra")
            return im

        def grid(t):
            im = canvas("G1  /  Four synchronized carries", f"PPO {iteration}     |     {t / 50:.2f} s")
            for k in range(4):
                im.paste(panel(k, t), (k % 2 * WIDTH, HEADER + k // 2 * HEIGHT))
            draw = ImageDraw.Draw(im)
            draw.line([(WIDTH, HEADER), (WIDTH, HEADER + 2 * HEIGHT)], fill=DARK, width=3)
            draw.line([(0, HEADER + HEIGHT), (2 * WIDTH, HEADER + HEIGHT)], fill=DARK, width=3)
            return im

        def opening(k):
            im = canvas(f"Box start   /   {k * 10} cm", "SAME ROBOT START     /     FIXED CAMERA")
            im.paste(panel(k, 0, True), (0, HEADER))
            return im

        if args.preview:
            opening(3).resize((768, 432)).save(args.preview)
            return
        with imageio.get_writer(
            args.output,
            fps=25,
            codec="libx264",
            macro_block_size=None,
            ffmpeg_params=["-threads", "3", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        ) as writer:
            for k in range(4):
                intro = np.asarray(opening(k))
                for _ in range(32):
                    writer.append_data(intro)
            grid(0).save(args.output.with_suffix(".jpg"), quality=92)
            for t in range(0, traces.shape[1], 2):
                writer.append_data(np.asarray(grid(t)))
            final = np.asarray(grid(traces.shape[1] - 1))
            for _ in range(12):
                writer.append_data(final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trials", "bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    render(parser.parse_args())
