"""Render the fitted bimanual reference in a legible three-quarter view."""

import argparse
import json
import subprocess
from pathlib import Path
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(dataset, bundle, dest):
    fit = json.loads((bundle / "fit.json").read_text())
    entry = next(
        x
        for x in json.loads((dataset / "accepted_metadata.json").read_text())["accepted"]
        if x["sequence"] == fit["sequence"]
    )
    ref = np.load(bundle / "motion.npz")
    spec = mujoco.MjSpec.from_file(str(dataset / f"g1-{dataset.name}.xml"))
    scale = entry["effective_mesh_scale"]
    spec.mesh("object_collision_mesh").scale[:] *= scale / spec.mesh("object_mesh").scale.copy()
    spec.mesh("object_mesh").scale[:] = scale
    spec.geom("object_visual").rgba = [0.84, 0.49, 0.19, 1]
    spec.add_texture(
        name="tiles",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=128,
        height=128,
        rgb1=[0.36, 0.41, 0.46],
        rgb2=[0.43, 0.48, 0.53],
    )
    mat = spec.add_material(name="floor", texrepeat=[3, 3], texuniform=True)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tiles"
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1], material="floor"
    )
    spec.worldbody.add_light(
        pos=[1, 1, 4], dir=[-0.2, -0.2, -1], ambient=[0.25] * 3, diffuse=[0.65] * 3, castshadow=True
    )
    spec.visual.global_.offwidth = 832
    spec.visual.global_.offheight = 592
    model = spec.compile()
    d = mujoco.MjData(model)
    ids = [list(ref["joint_names"]).index(model.joint(i).name) for i in range(1, model.njnt - 1)]
    rq = ref["qpos"]
    bp = ref["box_pose"]
    mid = bp[:, 2] > np.quantile(bp[:, 2], 0.5)
    facing = (bp[mid, :2] - rq[mid, :2]).mean(0)
    facing /= np.linalg.norm(facing)
    travel = bp[-1, :2] - bp[0, :2]
    # Front quarter from the open side of the carry path.
    side = np.array([-travel[1], travel[0]]) / np.linalg.norm(travel)
    if np.dot(side, facing) < 0:
        side *= -1
    view = side + 0.8 * facing
    cam = mujoco.MjvCamera()
    cam.azimuth = np.rad2deg(np.arctan2(-view[1], -view[0]))
    cam.elevation = -12
    cam.distance = 2.7
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    opt.sitegroup[:] = 0
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 18) if Path(font_path).exists() else ImageFont.load_default(size=18)
    n = len(bp)
    idx = np.unique(np.linspace(0, n - 1, 6).astype(int))
    sheet = Image.new("RGB", (3 * 416, 2 * 296))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (
        mujoco.Renderer(model, height=592, width=832) as renderer,
        subprocess.Popen(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "832x592",
                "-r",
                "50",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dest),
            ],
            stdin=subprocess.PIPE,
        ) as enc,
    ):
        for i in range(n):
            d.qpos[:] = np.r_[rq[i, :7], rq[i, 7:][ids], bp[i]]
            mujoco.mj_kinematics(model, d)
            mujoco.mj_forward(model, d)
            cam.lookat[:] = [*((0.6 * rq[i, :2] + 0.4 * bp[i, :2]) * 0.82 + 0.18 * (rq[:, :2].mean(0))), 0.65]
            renderer.update_scene(d, camera=cam, scene_option=opt)
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            draw.rounded_rectangle([14, 12, 417, 46], 6, fill=(27, 36, 44))
            draw.text(
                (23, 18),
                f"{fit['sequence']}   |   fitted reference   {i / 50:.1f} s",
                font=font,
                fill="white",
            )
            enc.stdin.write(img.tobytes())
            if i in idx:
                k = list(idx).index(i)
                sheet.paste(img.resize((416, 296)), ((k % 3) * 416, (k // 3) * 296))
        enc.stdin.close()
        if enc.wait():
            raise RuntimeError("ffmpeg failed")
    sheet.save(dest.with_suffix(".jpg"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["dataset", "bundle", "dest"]:
        p.add_argument(key, type=Path)
    a = p.parse_args()
    render(a.dataset, a.bundle, a.dest)
