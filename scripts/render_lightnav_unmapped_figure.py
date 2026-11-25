"""Render the unassisted LightNav stairs rollout using its logs and an archived command replay."""

import argparse
import gzip
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from build_lightnav_stairs_figure import INK, MUTED, PLAN, TRACK, read_rows

KEY_POSES = [
    (0.62, 0.45),
    (3.06, 0.54),
    (7.22, 0.69),
    (9.94, 0.72),
    (13.98, 0.77),
    (15.86, 0.84),
    (17.82, 0.91),
    (20.38, 1.0),
]
QUERY_TIME = 7.20
FOV = 89.42
# Pixel landmarks digitized in the archived navigation crop at t = 7.22 s.
# Near box (five corners), far door (two corners), end wall and three stair edges.
PIXELS = np.array(
    [
        [123.6, 100.6],
        [177.8, 103.6],
        [126.4, 145.6],
        [176.9, 159.8],
        [180.5, 97.8],
        [187.2, 60.8],
        [202.0, 59.5],
        [234.3, 47.8],
        [164.0, 58.6],
        [265.0, 87.2],
        [273.4, 93.0],
        [284.5, 99.7],
    ]
)
LANDMARKS = np.array(
    [
        [4.2, 1.63, 0.92],
        [4.2, 0.61, 0.92],
        [4.2, 1.63, 0],
        [4.2, 0.61, 0],
        [4.98, 0.61, 0.92],
        [10.63, 0.515, 2.96],
        [10.63, -0.515, 2.96],
        [10.69, -2.5, 3.44],
        [10.69, 2.5, 3.44],
        [7.15, -2, 0.9],
        [6.8, -2, 0.75],
        [6.45, -2, 0.60],
    ]
)


def camera_basis(angles):
    sy, sp, sr = np.sin(angles)
    cy, cp, cr = np.cos(angles)
    right = np.array([sy, -cy, 0])
    up = np.array([-cy * sp, -sy * sp, cp])
    forward = np.array([cy * cp, sy * cp, sp])
    return np.stack((right * cr + up * sr, up * cr - right * sr, -forward), axis=1)


def project(points, pos, rotation, fovy, width, height):
    view = (points - pos) @ rotation
    focal = (height / 2) / np.tan(np.deg2rad(fovy / 2))
    return np.array([width / 2 - 0.5, height / 2 - 0.5]) - focal * view[..., :2] * [1, -1] / view[..., 2:]


def register_navigation_camera(query, pelvis):
    def residual(pose):
        return (project(LANDMARKS, pose[:3], camera_basis(pose[3:]), FOV, 304, 174) - PIXELS).ravel()

    initial = np.r_[pelvis + [0.12, 0.05, 0.45], query["capture_yaw"], 0, 0]
    fit = least_squares(residual, initial, loss="soft_l1", f_scale=0.7)
    pixel_errors = np.linalg.norm(residual(fit.x).reshape(-1, 2), axis=1)
    if pixel_errors.max() > 3:
        raise ValueError("Navigation-camera registration exceeded three archived pixels.")
    return fit.x, pixel_errors


def style_scene(m, origin):
    m.vis.global_.offwidth = 2880
    m.vis.global_.offheight = 1100
    m.stat.center[:] = origin + [4, 0, 1]
    m.stat.extent = 8
    # Recenter the infinite plane's rendered patch; its physical plane is unchanged.
    floor = m.geom("terrain_0")
    # At this tile's large world offset MuJoCo clips the drawn infinite-plane mesh.
    # A display-only slab provides the same z=0 surface across the entire visible room.
    floor.type[0] = mj.mjtGeom.mjGEOM_BOX
    floor.pos[:] = origin + [4, 0, -0.05]
    floor.size[:] = [90, 90, 0.05]
    m.geom_rbound[floor.id] = np.linalg.norm(floor.size)
    colors = {
        "station_wall": (0.77, 0.80, 0.80),
        "station_end": (0.72, 0.76, 0.76),
        "station_wainscot": (0.49, 0.59, 0.60),
        "landing_door": (0.11, 0.32, 0.26),
        "terrain_0": (0.80, 0.82, 0.81),
        "nav_stack": (0.57, 0.41, 0.25),
        "nav_crate": (0.57, 0.41, 0.25),
        "nav_barricade": (0.81, 0.34, 0.17),
        "terrain_": (0.60, 0.54, 0.44),
        "window_glass": (0.64, 0.76, 0.78),
        "window_frame": (0.32, 0.40, 0.42),
        "window_mullion": (0.32, 0.40, 0.42),
    }
    for i in range(m.ngeom):
        name = m.geom(i).name
        for prefix, color in colors.items():
            if name.startswith(prefix):
                m.geom_rgba[i, :3] = color
                break
    # Reuse the compiled model's unused ground texture for a fine-grained wood finish.
    tex = m.tex("groundplane").id
    height, width = int(m.tex_height[tex]), int(m.tex_width[tex])
    y, x = np.mgrid[:height, :width]
    grain = np.sin(
        2
        * np.pi
        * (27 * y / height + 0.32 * np.sin(2 * np.pi * x / width) + 0.15 * np.sin(6 * np.pi * x / width))
    )
    value = np.clip(225 + 9 * grain + 4 * np.cos(58 * np.pi * y / height), 0, 255).astype(np.uint8)
    channels = int(m.tex_nchannel[tex])
    texture = np.repeat(value[:, :, None], channels, axis=2).ravel()
    adr = int(m.tex_adr[tex])
    m.tex_data[adr : adr + len(texture)] = texture
    wood = m.mat("groundplane").id
    m.mat_rgba[wood] = [0.66, 0.48, 0.31, 1]
    m.mat_texuniform[wood] = False
    m.mat_texrepeat[wood] = [1.0, 1.0]
    m.mat_reflectance[wood] = 0
    m.mat_specular[wood] = 0.04
    for name in ("nav_crate", "nav_stack"):
        m.geom_matid[m.geom(name).id] = wood
    m.vis.headlight.ambient[:] = 0.43
    m.vis.headlight.diffuse[:] = [0.26, 0.28, 0.30]
    m.vis.headlight.specular[:] = 0.20
    m.light_active[:] = False
    lid = m.light("station_daylight").id
    m.light_active[lid] = True
    m.light_pos[lid] = origin + [2, -4, 8]
    m.light_dir[lid] = [0.11, 0.42, -1]
    m.light_diffuse[lid] = [0.70, 0.67, 0.62]
    m.light_ambient[lid] = 0.035
    m.light_specular[lid] = 0.15
    m.light_castshadow[lid] = True
    m.vis.quality.shadowsize = 4096
    m.vis.map.shadowclip = 1.1


def add_curve(scene, points, color, radius, dots=False):
    identity = np.eye(3).ravel()
    rgba = np.array([*color, 1], dtype=np.float32)
    for i in range(1, len(points)):
        g = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), points[i], identity, rgba)
        mj.mjv_connector(g, mj.mjtGeom.mjGEOM_CAPSULE, radius, points[i - 1], points[i])
        g.emission = 0.30
        g.specular = 0
        if dots:
            g = scene.geoms[scene.ngeom]
            scene.ngeom += 1
            mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.032), points[i], identity, rgba)
            g.emission = 0.33


def on_surface(xy, tile, zoff):
    heights = np.r_[tile["base"], tile["base"] + np.cumsum(tile["jumps"])]
    return np.column_stack([xy, heights[np.searchsorted(tile["edges"], xy[:, 0], side="right")] + zoff])


def camera(m, origin, pos, rotation, fovy):
    cid = m.cam("review_camera").id
    m.cam_pos[cid] = origin + pos
    m.cam_quat[cid] = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    m.cam_fovy[cid] = fovy


def render_views(m, replay, summary, xyz, motion, query):
    origin = replay["origin"].astype(float)
    qpos = replay["qpos"]
    if len(qpos) != len(xyz) or not np.allclose(
        qpos[:, :3] - origin, np.vstack([xyz[0], replay["measurements"][:, :3]]), atol=0.001
    ):
        raise ValueError("Incompatible camera replay or root coordinate frame.")
    if not np.isclose(m.geom("nav_stack").pos[0] - origin[0], summary["fixtures"][2]["x"], atol=0.001):
        raise ValueError("Scene and rollout do not match.")
    style_scene(m, origin)
    data = mj.MjData(m)
    opt = mj.MjvOption()
    opt.sitegroup[:] = 0
    opt.geomgroup[:] = [1, 0, 0, 0, 0, 0]
    tile = summary["tile"]
    track = on_surface(xyz[:, :2], tile, 0.018)
    risers = np.nonzero(np.diff(track[:, 2]) != 0)[0]
    selected = np.unique(np.r_[np.arange(0, len(track), 3), risers, risers + 1, len(track) - 1])
    track = track[selected] + origin
    waypoints = on_surface(np.array(query["plan_xy"]), tile, 0.035) + origin

    def paint(renderer, show_track, sky_mask=None):
        mj.mj_forward(m, data)
        renderer.update_scene(data, camera="review_camera", scene_option=opt)
        renderer.scene.flags[mj.mjtRndFlag.mjRND_FOG] = False
        if sky_mask is None:
            renderer.enable_depth_rendering()
            sky_mask = renderer.render() > 100
            renderer.disable_depth_rendering()
        if show_track:
            add_curve(renderer.scene, track, (0.015, 0.33, 0.36), 0.011)
        add_curve(renderer.scene, waypoints, (0.83, 0.27, 0.055), 0.013, dots=True)
        img = renderer.render()
        img[sky_mask & (img.max(axis=2) < 10)] = [219, 229, 232]
        return img, sky_mask

    # A shared static spectator camera for every exposure.
    eye = np.array([0.5, -13.3, 5.50])
    direction = np.array([4.65, 0.0, 1.08]) - eye
    direction /= np.linalg.norm(direction)
    yaw = np.arctan2(direction[1], direction[0])
    camrot = camera_basis([yaw, np.arcsin(direction[2]), 0])
    hero_wh = (2880, 1040)
    plan_anchor = project(waypoints[3] - origin, eye, camrot, 17.5, *hero_wh)
    camera(m, origin, eye, camrot, 17.5)
    errors = {}
    with mj.Renderer(m, width=hero_wh[0], height=hero_wh[1]) as renderer:
        data.qpos[:] = qpos[0]
        base, sky_mask = paint(renderer, True)
        composite = base.copy()
        for t, alpha in KEY_POSES:
            idx = round(t * 50)
            row = motion[idx - 1]
            if abs(row["t"] - t) > 0.0001:
                raise ValueError("Missing recorded pose")
            q = qpos[idx].astype(float).copy()
            offset = xyz[idx] - (q[:3] - origin)
            q[:3] += offset
            data.qpos[:] = q
            opt.geomgroup[2] = 1
            pose, _ = paint(renderer, True, sky_mask)
            changed = np.any(pose != base, axis=2)
            composite[changed] = np.rint((1 - alpha) * composite[changed] + alpha * pose[changed]).astype(
                "uint8"
            )
            err_foot = replay["measurements"][idx - 1, 3:5] + offset[0] - row["feet_x"]
            errors[f"{t:.2f}"] = dict(
                root_shift_m=offset.tolist(), aligned_foot_x_error_m=err_foot.tolist(), opacity=alpha
            )
    nav_pose, err = register_navigation_camera(query, xyz[round(7.22 * 50)])
    camera(m, origin, nav_pose[:3], camera_basis(nav_pose[3:]), FOV)
    opt.geomgroup[2] = 0
    with mj.Renderer(m, width=1824, height=1044) as renderer:
        nav, _ = paint(renderer, False)
    qa = {
        "pose_reconstruction": errors,
        "nav_camera_local_xyz": nav_pose[:3].tolist(),
        "nav_camera_rpy_values_yaw_pitch_roll_rad": nav_pose[3:].tolist(),
        "nav_landmark_reprojection_error_px": err.tolist(),
        "full_command_replay_max_root_error_m": float(
            np.max(np.linalg.norm(replay["measurements"][:, :3] - xyz[1:], axis=1))
        ),
        "plan_anchor_hero_px": plan_anchor.tolist(),
        "hero_size_px": list(hero_wh),
        "nav_size_px": [1824, 1044],
    }
    # Normalized top-down plot and the native renderer share the logged route.
    return composite, nav, qa


def build(args):
    exp = args.experiment
    source = exp / "evaluation/legacy3800_online_far_door"
    prov = exp / "provenance/unmapped_hd_figure"
    out = exp / "results/unmapped_hd_figure"
    out.mkdir(parents=True, exist_ok=True)
    (out / "qa").mkdir(exist_ok=True)
    summary = json.loads((source / "result.json").read_text())
    motion = read_rows(source / "motion.jsonl")
    preds = read_rows(source / "predictions.jsonl")
    if summary["outcome"] != "arrived" or summary.get("clearance_guide_m") or not summary["wide_nav_camera"]:
        raise ValueError("Expected the archived unmapped dual-camera arrival.")
    query = next(p for p in preds if np.isclose(p["sim_time"], QUERY_TIME))
    xyz = np.array([summary["start"]] + [r["xyz"] for r in motion])
    with gzip.open(prov / "scene.mjb.gz", "rb") as fh:
        model = mj.MjModel.from_binary_path("scene.mjb", assets={"scene.mjb": fh.read()})
    with np.load(prov / "command_replay.npz") as replay:
        hero, nav, meta = render_views(model, replay, summary, xyz, motion, query)
    obj = next(o for o in summary["fixtures"] if o["name"] == "stack")
    ox, oy = query["capture_xy"]
    meta.update(
        source=source.name,
        query_time_s=query["sim_time"],
        camera_film_time_s=7.22,
        target_visible_output=query["response"]["visible"],
        body_to_crate_deg=float(np.degrees(np.arctan2(obj["y"] - oy, obj["x"] - ox) - query["capture_yaw"])),
    )
    (out / "qa/render_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    draw_figure(out, hero, nav, summary, xyz, query, meta, args.qa_scripts)


def draw_figure(out, hero, nav, summary, xyz, query, meta, qa_scripts):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "text.color": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    w, h = 180, 131
    fig = plt.figure(figsize=(w / 25.4, h / 25.4), facecolor="white")

    def area(x, y, width, height):
        ax = fig.add_axes([x / w, 1 - (y + height) / h, width / w, height / h])
        ax.set_axis_off()
        return ax

    def label(x, y, s, **kwargs):
        fig.text(x / w, 1 - y / h, s, va="center", **kwargs)

    def title(x, y, letter, txt):
        label(x, y, letter, fontsize=9.2, weight="bold")
        label(x + 5, y, txt, fontsize=8.4, weight="bold")

    title(3, 4, "a", "Language-guided stair ascent")
    label(177, 4, "LightNav-0  /  DA2 3800", ha="right", fontsize=7.2, color=MUTED)
    label(3, 10, "“" + summary["instruction"] + "”", fontsize=8.0, style="italic")
    overview = area(3, 14, 174, 174 * hero.shape[0] / hero.shape[1])
    overview.imshow(hero, interpolation="antialiased", aspect="auto")
    ax, ay = meta["plan_anchor_hero_px"]
    overview.annotate(
        "Local plan at 7.20 s",
        xy=(ax, ay + 4),
        xytext=(ax - 155, ay + 125),
        ha="center",
        fontsize=6.3,
        color="#71350c",
        arrowprops=dict(arrowstyle="-", color="#925327", lw=0.55),
    )
    title(3, 82, "b", "Route past the second crate")
    title(103, 82, "c", "Crate and doorway in view")
    label(3, 87, "7.20 s", fontsize=6.2, color=MUTED)
    label(103, 87, "Navigation camera at 7.22 s  ·  reconstructed view", fontsize=6.2, color=MUTED)
    # Match the lower panel heights and keep equal metric scales in the plan.
    plan = area(3, 91, 95, 35)
    left, right = -0.50, 9.15
    hh = (right - left) * 35 / 95 / 2
    plan.set_xlim(left, right)
    plan.set_ylim(-hh, hh)
    tile = summary["tile"]
    plan.add_patch(Rectangle((tile["start"], -hh), right - tile["start"], 2 * hh, fc="#ebece6", ec="none"))
    for edge in tile["edges"]:
        plan.plot([edge, edge], [-hh, hh], lw=0.5, color="#bcc6c1")
    for o in summary["fixtures"]:
        width, height = o["length"], o["width"]
        plan.add_patch(
            Rectangle(
                (o["x"] - width / 2, o["y"] - height / 2),
                width,
                height,
                fc="#d8b895" if o["style"] == "crate" else "#dea483",
                ec="#967e65",
                lw=0.5,
            )
        )
    plan.plot(*xyz[:, :2].T, color=TRACK, lw=1.55, zorder=3, solid_capstyle="round")
    wp = np.array(query["plan_xy"])
    px, py = wp[0]
    yaw = query["capture_yaw"]
    plan.annotate(
        "",
        xy=wp[0] + 1.22 * np.array([np.cos(yaw), np.sin(yaw)]),
        xytext=(px, py),
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.85, shrinkA=0, shrinkB=0),
        zorder=4,
    )
    plan.plot(*wp.T, color=PLAN, lw=0.9, zorder=5)
    plan.scatter(*wp[1:].T, fc=PLAN, ec="white", lw=0.25, s=5, zorder=6)
    plan.scatter(px, py, fc="white", ec=PLAN, s=17, lw=0.9, zorder=7)
    plan.plot(*summary["goal_xy"], marker="D", ms=3.7, mfc="white", mec=INK, mew=0.8)
    plan.text(8.12, -0.47, "Landing goal", fontsize=5.6, ha="center", color=MUTED)
    plan.plot([-0.15, 0.85], [-1.29, -1.29], lw=0.9, color=INK)
    plan.text(0.35, -1.12, "1 m", ha="center", fontsize=5.4)
    fig.legend(
        [
            Line2D([], [], color=TRACK, lw=1.5),
            Line2D([], [], color=PLAN, marker="o", ms=2, lw=0.8),
            Line2D([], [], color=INK, marker=">", markevery=[1], lw=0.8, ms=3),
        ],
        ["Executed", "Local plan", "Body heading"],
        loc="center left",
        bbox_to_anchor=(18 / w, 1 - 87 / h),
        ncol=3,
        frameon=False,
        fontsize=5.8,
        handlelength=1.5,
        columnspacing=0.85,
        borderaxespad=0,
    )
    ego = area(103, 91, 74, 35)
    # Crop only display margins; show the full near box and the far doorway.
    ego.imshow(nav, extent=(-0.5, 303.5, 173.5, -0.5), aspect="auto", interpolation="antialiased")
    ego.set_xlim(40, 304)
    ego.set_ylim(171, 171 - 264 * 35 / 74)
    ego.annotate(
        "Doorway",
        xy=(195, 67),
        xytext=(246, 61),
        ha="center",
        fontsize=6.4,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=2, shrinkB=2),
    )
    ego.annotate(
        "Second crate",
        xy=(173, 140),
        xytext=(231, 151),
        ha="center",
        fontsize=6.4,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=2, shrinkB=2),
    )
    label(
        3,
        129,
        "No obstacle-map guidance  ·  Contact with both crates  ·  Upper landing reached at 20.38 s",
        fontsize=6.1,
        color=MUTED,
    )
    if qa_scripts is not None:
        sys.path.insert(0, str(qa_scripts))
        from audit_panel_alignment import require_matplotlib_panel_alignment

        require_matplotlib_panel_alignment(
            fig,
            strict=True,
            json_out=out / "qa/alignment.json",
            panel_ids=["overview", "plan", "navigation"],
            row_groups=[["plan", "navigation"]],
            column_groups=[],
        )
    fig.savefig(out / "lightnav_unmapped_hd.pdf", dpi=420)
    fig.savefig(out / "lightnav_unmapped_hd.svg", dpi=420)
    fig.savefig(out / "lightnav_unmapped_hd.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--qa-scripts", type=Path, help="Optional directory containing audit_panel_alignment.py")
    build(parser.parse_args())
