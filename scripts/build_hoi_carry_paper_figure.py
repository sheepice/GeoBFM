"""Build the G1 carry plate from the original paired PPO-2500 evaluation trajectories."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import mujoco
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull

from render_hoi_pickup_offsets import scene_model

CASES = ("far30_left30", "far30", "far30_right30")
COLORS = ("#087f81", "#af6b1c", "#635baa")
INK, GRAY = "#192f38", "#4c626a"
W, H = 2640, 1520
CONTACT_TIME = 2.80
EXPOSURES = ((1.20, 0.58), (CONTACT_TIME, 0.68), (3.98, 0.79), (6.40, 1.0))


def get_runs(folder, bundle):
    runs = []
    identities = []
    sources = []
    with np.load(bundle / "motion.npz") as ref:
        forward = ref["box_pose"][0, :2] - ref["qpos"][0, :2]
        forward /= np.linalg.norm(forward)
        box0 = ref["box_pose"][0, :2]
        goal = ref["box_pose"][-1, :2]
    left = np.array([-forward[1], forward[0]])
    for case, lateral in zip(CASES, (0.30, 0, -0.30), strict=True):
        path = folder / case / "closed_loop.npz"
        meta = json.loads((folder / case / "evaluation.json").read_text())
        with np.load(path) as data:
            poses = data["qpos"].copy()
            assert int(data["fps"]) == 50 and np.isfinite(poses).all()
            if case == "far30":
                assert (data["hand_force"][60] > 0.5).all() and data["floor_force"][60] > 1
                assert data["box_bottom"][100] > 0.15 and data["floor_force"][100] < 0.5
                assert (data["hand_force"][210] > 0.5).all()
                assert max(data["hand_force"][320]) < 0.5 and data["floor_force"][320] > 1
        assert meta["iteration"] == 2500 and meta["selected"]["env"] == 0 and meta["selected"]["placed"]
        assert meta["vision"] == "normal"
        identities.append((meta["checkpoint_sha256"], meta["initial_robot_qpos_sha256"], meta["seed"]))
        np.testing.assert_allclose(poses[0, -7:-5], box0 + 0.30 * forward + lateral * left, atol=1e-5)
        if runs:
            np.testing.assert_array_equal(poses[0, :-7], runs[0][0, :-7])
        runs.append(poses)
        sources.append(
            dict(case=case, replay=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        )
    assert len(set(identities)) == 1
    return runs, forward, left, goal, sources


def project(scene, point):
    a, b = scene.camera
    eye = (a.pos + b.pos) / 2
    direction = a.forward / np.linalg.norm(a.forward)
    right = np.cross(direction, a.up)
    right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    v = np.asarray(point) - eye
    focal = H * a.frustum_near / (a.frustum_top - a.frustum_bottom)
    return np.array(
        [W / 2 + focal * (v @ right) / (v @ direction), H / 2 - focal * (v @ up) / (v @ direction)]
    )


def pose_images(bundle, poses):
    model = scene_model(bundle)
    model.vis.global_.offwidth = W
    model.vis.global_.offheight = H
    floor = model.geom("floor").id
    # Reserve a render group for the floor so a floor-only pass gives a paired shadow reference.
    model.geom_group[floor] = 5
    model.vis.headlight.ambient[:] = 0.55
    model.vis.headlight.diffuse[:] = 0.28
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 87, -14, 3.65
    cam.lookat[:] = [0.26, -0.56, 0.68]
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 1]
    opt.sitegroup[:] = 0
    empty = mujoco.MjvOption()
    empty.geomgroup[:] = [0, 0, 0, 0, 0, 1]
    empty.sitegroup[:] = 0
    exposures = {}
    with mujoco.Renderer(model, height=H, width=W) as renderer:
        data.qpos[:] = poses[0]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam, scene_option=empty)
        base = renderer.render().mean(-1, dtype=np.float32)
        for t, _ in EXPOSURES:
            data.qpos[:] = poses[round(t * 50)]
            mujoco.mj_forward(model, data)
            if t == CONTACT_TIME:
                thigh = model.geom("right_thigh_collision").id
                box = model.geom("box/object_collision").id
                gap = mujoco.mj_geomDistance(model, data, thigh, box, 0.1, None)
                assert gap < 0, f"Expected right thigh contact at {t:.2f} s; gap {gap:.4f} m"
            renderer.update_scene(data, camera=cam, scene_option=opt)
            rgb = renderer.render()
            renderer.enable_segmentation_rendering()
            seg = renderer.render()
            renderer.disable_segmentation_rendering()
            visible = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            obj = visible & (seg[..., 0] != floor)
            shade = np.where(
                visible & ~obj, np.maximum(0, base - rgb.mean(-1, dtype=np.float32)) / np.maximum(base, 12), 0
            )
            shade = np.clip(gaussian_filter(shade, 1.0) * 1.1, 0, 0.55)
            # A local floor shadow grounds the feet without accumulating distant floor reflections.
            feet = [model.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
            centers = [project(renderer.scene, [*data.xpos[f, :2], 0]) for f in feet]
            if poses[round(t * 50), -5] < 0.24:
                centers.append(project(renderer.scene, [*poses[round(t * 50), -7:-5], 0]))
            xxgrid = np.arange(W)[None, :]
            yygrid = np.arange(H)[:, None]
            envelope = np.maximum.reduce(
                [
                    np.exp(-0.5 * ((xxgrid - x) / 120) ** 2 - 0.5 * ((yygrid - y) / 30) ** 2)
                    for x, y in centers
                ]
            )
            shade *= envelope
            rgba = np.zeros((H, W, 4), dtype=np.uint8)
            rgba[:, :, :3] = [105, 116, 119]
            rgba[:, :, 3] = np.rint(shade * 255).astype(np.uint8)
            rgba[obj, :3] = rgb[obj]
            rgba[obj, 3] = 255
            yy, xx = np.where(obj)
            bb = (int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1))
            exposures[t] = (Image.fromarray(rgba), bb)
        # Ground-projected skeleton origin and box center supply a metric, world-registered track.
        ground = np.column_stack((poses[:, :2], np.full(len(poses), 0.008)))
        line = np.stack([project(renderer.scene, p) for p in ground])
    visual = model.geom("box/object_visual").id
    meshid = model.geom_dataid[visual]
    vertices = model.mesh_vert[
        model.mesh_vertadr[meshid] : model.mesh_vertadr[meshid] + model.mesh_vertnum[meshid]
    ]
    data.qpos[:] = poses[0]
    mujoco.mj_kinematics(model, data)
    world = vertices @ data.geom_xmat[visual].reshape(3, 3).T + data.geom_xpos[visual]
    footprint = world[ConvexHull(world[:, :2]).vertices, :2] - poses[0, -7:-5]
    return exposures, line, footprint


def faded(image, opacity):
    if opacity == 1:
        return image
    out = image.copy()
    out.putalpha(image.getchannel("A").point(lambda value: round(value * opacity)))
    return out


def figure(args):
    args.output.mkdir(parents=True, exist_ok=True)
    qa = args.output / "qa"
    qa.mkdir(exist_ok=True)
    runs, forward, left, goal, sources = get_runs(args.trials, args.bundle)
    poses = runs[1]
    images, path, footprint = pose_images(args.bundle, poses)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.55,
            "text.color": INK,
            "axes.labelcolor": GRAY,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "savefig.facecolor": "white",
        }
    )
    width, height = 180, 63
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), facecolor="white")

    def axes(x, y, w, h):
        return fig.add_axes([x / width, y / height, w / width, h / height])

    def label(x, y, string, size=7, **kwargs):
        return fig.text(x / width, y / height, string, fontsize=size, va="baseline", **kwargs)

    def heading(x, y, letter, string):
        label(x, y, letter, 10, weight="bold")
        label(x + 5.2, y + 0.1, string, 9, weight="bold")

    heading(2, 57, "a", "Whole-body box transport")
    heading(115, 57, "b", "Box starts and pelvis paths")
    hero = axes(1.5, 7, 108.5, 47)
    hero.axis("off")
    boxes = np.array([images[t][1] for t, _ in EXPOSURES])
    bounds = (boxes[:, 0].min() - 48, boxes[:, 1].min() - 48, boxes[:, 2].max() + 48, boxes[:, 3].max() + 52)
    x0, y0, x1, y1 = bounds
    # Adjust the viewing window, never the coordinates of individual poses.
    wanted = 108.5 / 47
    dw = max(0, (y1 - y0) * wanted - (x1 - x0))
    dh = max(0, (x1 - x0) / wanted - (y1 - y0))
    x0 -= dw / 2
    x1 += dw / 2
    y0 -= dh / 2
    y1 += dh / 2
    hero.set(xlim=(x0, x1), ylim=(y1, y0))
    hero.set_facecolor("white")
    hero.plot(path[:, 0], path[:, 1], lw=0.95, color="#508f95", alpha=0.75, solid_capstyle="round", zorder=1)
    combined = Image.new("RGBA", (W, H))
    for t, opacity in EXPOSURES:
        combined.alpha_composite(faded(images[t][0], opacity))
    hero.imshow(
        combined, extent=(-0.5, W - 0.5, H - 0.5, -0.5), zorder=2, aspect="auto", interpolation="antialiased"
    )
    hero.set(xlim=(x0, x1), ylim=(y1, y0))

    # Rotate into a metric robot-start frame and use a single scale in both directions.
    origin = poses[0, :2]

    def planar(xy):
        delta = xy - origin
        return np.stack([delta @ (-forward), delta @ (-left)], axis=-1)

    plan = axes(116, 7, 61, 47)
    plan.set_aspect("equal", adjustable="box")
    plan.axis("off")
    for run, col, style in zip(runs, COLORS, ["-", "--", ":"], strict=True):
        xy = planar(run[:, :2])
        plan.plot(*xy.T, color=col, linewidth=1.15, linestyle=style, solid_capstyle="round")
    for run, col, name in zip(runs, COLORS, ("Left", "Center", "Right"), strict=True):
        xy = planar(run[0, -7:-5])
        polygon = planar(run[0, -7:-5] + footprint)
        plan.fill(*polygon.T, facecolor=col, edgecolor="none", alpha=0.16, zorder=2)
        plan.plot(*np.vstack([polygon, polygon[:1]]).T, color=col, lw=0.80, zorder=4)
        plan.plot(*xy, marker=".", ms=2.5, color=col, zorder=5)
        plan.annotate(
            name,
            xy,
            xytext=(-15, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=6.5,
            color=col,
        )
    goalxy = planar(goal)
    plan.add_patch(Circle(goalxy, 0.15, edgecolor="#7c8887", facecolor="#eff2ef", lw=0.6, zorder=0))
    plan.plot(*goalxy, marker="+", color="#344e50", ms=5, mew=1)
    plan.annotate("Place", goalxy, xytext=(2, 12), textcoords="offset points", ha="center", fontsize=6.2)
    plan.plot(0, 0, "o", color="white", markeredgecolor=INK, ms=4, zorder=6)
    plan.annotate("Robot start", (0, 0), xytext=(8, 3), textcoords="offset points", fontsize=6.2)
    plan.plot([-0.08, 0.42], [0.42, 0.42], color=INK, lw=1, solid_capstyle="butt")
    plan.text(0.17, 0.58, "0.5 m", ha="center", va="center", fontsize=6.1)
    # Fit the metric plan to the full row while preserving its original center and scale.
    xlim = (-1.84, 1.37)
    half_height = (xlim[1] - xlim[0]) * 47 / (2 * 61)
    plan.set(xlim=xlim, ylim=(-0.305 - half_height, -0.305 + half_height))
    label(3, 4.2, "Center pickup", 6.4, color=COLORS[1], weight="bold")
    label(23.5, 4.2, "Later poses appear darker.", 6.4, color=GRAY)
    label(115, 4.2, "Box centers are spaced 30 cm apart.", 6.4, color=GRAY)
    if args.qa_tools is not None:
        sys.path.insert(0, str(args.qa_tools))
        from audit_panel_alignment import require_matplotlib_panel_alignment

        require_matplotlib_panel_alignment(
            fig,
            axes=[hero, plan],
            panel_ids=["a", "b"],
            row_groups=[dict(id="top-row", panels=["a", "b"])],
            json_out=qa / "alignment.json",
            strict=True,
        )
    basename = "hoi_box_transport"
    fig.savefig(args.output / (basename + ".pdf"), dpi=600)
    fig.savefig(args.output / (basename + ".png"), dpi=600)
    fig.savefig(args.output / (basename + ".svg"), dpi=600)
    with Image.open(args.output / (basename + ".png")) as source:
        source.thumbnail((1200, 850), Image.Resampling.LANCZOS)
        source.save(qa / "preview.png")
    plt.close(fig)
    (qa / "provenance.json").write_text(
        json.dumps(
            dict(
                cases=sources,
                checkpoint=2500,
                selected_env=0,
                seed=20261125,
                capture_rate_hz=50,
                exposure_times_opacity=EXPOSURES,
                camera_azimuth_elevation=[87, -14],
                width_mm=width,
                height_mm=height,
                rendering="Original MuJoCo mesh and replay coordinates. Masked visual geometry and cast shadows on a white background. The four poses share a fixed camera.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("trials", "bundle", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--qa-tools", type=Path, help="Optional directory containing audit_panel_alignment.py")
    figure(parser.parse_args())
