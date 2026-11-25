"""Render reference and two recorded physical carry trials with an identical camera and clock."""

import argparse
from pathlib import Path

import _bootstrap
import imageio.v2 as imageio
import mujoco
from mjlab.scene import Scene
import numpy as np
from PIL import Image, ImageDraw

from da2_vibe.paths import load_settings
from hoi_carry_task import config


def render(bundle, old_file, new_file, output):
    with np.load(bundle / "motion.npz") as ref, np.load(old_file) as old, np.load(new_file) as new:
        target = np.column_stack([ref["qpos"], ref["box_pose"]])
        old_q = old["qpos"]
        new_q = new["qpos"]
        assert int(old["fps"]) == int(new["fps"]) == int(ref["fps"]) == 50
    duration = max(len(old_q), len(new_q))
    phases = np.clip(np.arange(duration) - 1, 0, len(target) - 1)
    settings = load_settings(str(_bootstrap.ROOT / "configs/tasks/teleop.json"))
    model = Scene(config(bundle, settings, 1, 20261125, evaluate=True).scene, "cpu").compile()
    assert model.nq == target.shape[1] == old_q.shape[1] == new_q.shape[1]
    model.vis.global_.offwidth = 480
    model.vis.global_.offheight = 400
    model.vis.headlight.ambient[:] = 0.48
    model.vis.headlight.diffuse[:] = 0.72
    data = mujoco.MjData(model)
    opt = mujoco.MjvOption()
    opt.geomgroup[3:] = 0
    opt.sitegroup[:] = 0
    cam = mujoco.MjvCamera()
    cam.elevation = -12
    cam.distance = 2.8
    box = target[:, 36:39]
    pelvis = target[:, :3]
    travel = box[-1, :2] - box[0, :2]
    lateral = np.array([-travel[1], travel[0]]) / np.linalg.norm(travel)
    facing = (box - pelvis)[:, :2][box[:, 2] > np.median(box[:, 2])].mean(0)
    facing /= np.linalg.norm(facing)
    if np.dot(lateral, facing) < 0:
        lateral *= -1
    view = lateral + 0.8 * facing
    cam.azimuth = np.degrees(np.arctan2(-view[1], -view[0]))
    labels = ["Motion reference", "Previous HOI", "HOI from original SONIC"]
    with (
        mujoco.Renderer(model, height=400, width=480) as renderer,
        imageio.get_writer(
            output,
            fps=50,
            codec="libx264",
            ffmpeg_params=["-crf", "18", "-threads", "2", "-movflags", "+faststart"],
        ) as writer,
    ):
        for frame, phase in enumerate(phases):
            center = 0.6 * pelvis[phase, :2] + 0.4 * box[phase, :2]
            cam.lookat[:] = [*center, 0.65]
            sheet = Image.new("RGB", (1440, 432), "#1b252d")
            draw = ImageDraw.Draw(sheet)
            for col, seq in enumerate((target, old_q, new_q)):
                t = phase if col == 0 else frame
                if t < len(seq):
                    data.qpos[:] = seq[t]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=cam, scene_option=opt)
                    sheet.paste(Image.fromarray(renderer.render()), (480 * col, 32))
                else:
                    draw.text((480 * col + 130, 205), "Episode ended", fill="white", font_size=19)
                draw.text((480 * col + 12, 7), labels[col], font_size=17, fill="white")
            draw.text((1350, 7), f"{frame / 50:.2f} s", font_size=17, fill="white")
            writer.append_data(np.asarray(sheet))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "old", "new", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    render(a.bundle, a.old, a.new, a.output)
