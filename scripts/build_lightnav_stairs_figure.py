"""Compose the LightNav stairs demonstration from the archived film and synchronized logs."""

import argparse
import itertools
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes, median_filter, gaussian_filter

INK = "#182d33"
MUTED = "#486069"
TRACK = "#087f87"
PLAN = "#b65014"
# Capture times in the film; the final frame is written separately on arrival.
EXPOSURES = [
    (0.38, 0.43),
    (5.06, 0.57),
    (8.86, 0.56),
    (13.46, 0.65),
    (18.34, 0.71),
    (20.90, 0.82),
    (23.34, 0.90),
]
QUERIES = [5.04, 15.36, 21.60]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def review_camera(tile):
    """Reconstruct the recording camera in tile coordinates (add_review_camera)."""
    bounds = ([-0.15, -1, 0], [tile["goal"] + 0.15, 1, tile["total_height"] + 1.2])
    center = np.mean(bounds, axis=0)
    back = np.array([-0.15, -1.0, 0.35])
    back /= np.linalg.norm(back)
    right = np.cross([0, 0, 1], back)
    right /= np.linalg.norm(right)
    basis = np.stack([right, np.cross(back, right), back], axis=1)
    corners = (np.array(list(itertools.product(*zip(*bounds)))) - center) @ basis
    tangent = np.tan(np.deg2rad(43 / 2))
    distance = 1.04 * max(
        np.max(abs(corners[:, 0]) / (tangent * 640 / 360) + corners[:, 2]),
        np.max(abs(corners[:, 1]) / tangent + corners[:, 2]),
    )
    eye = center + distance * back

    def project(points):
        view = (np.asarray(points) - eye) @ basis
        scale = 180 / (-view[..., 2:3] * tangent)
        return [320, 180] + view[..., :2] * scale * [1, -1]

    return eye, project


def track_on_terrain(xyz, summary):
    tile = summary["tile"]
    edges = np.array(tile["edges"])
    levels = np.r_[tile["base"], tile["base"] + np.cumsum(tile["jumps"])]
    xyz = np.column_stack([xyz[:, :2], levels[np.searchsorted(edges, xyz[:, 0], side="right")] + 0.018])
    eye, project = review_camera(tile)
    rays = xyz - eye
    # Hide line segments behind solid fixtures and the camera-facing edges of the steps.
    boxes = []
    for obj in summary["fixtures"]:
        x, y = obj["x"], obj["y"]
        dx, dy = obj["length"] / 2, obj["width"] / 2
        boxes.append(([x - dx, y - dy, 0], [x + dx, y + dy, obj["height"]]))
    for i, edge in enumerate(edges):
        end = edges[i + 1] if i + 1 < len(edges) else tile["goal"] + 1.0
        boxes.append(([edge, -2, 0], [end, 2, levels[i + 1]]))
    visible = np.ones(len(xyz), bool)
    for lower, upper in boxes:
        with np.errstate(divide="ignore"):
            a, b = (np.array(lower) - eye) / rays, (np.array(upper) - eye) / rays
        near = np.maximum(np.min(np.stack([a, b]), axis=0).max(axis=1), 0.0)
        far = np.minimum(np.max(np.stack([a, b]), axis=0).min(axis=1), 0.99999)
        visible &= near > far
    uv = project(xyz)
    uv[~visible] = np.nan
    return uv


def get_foreground(frame, plate, point):
    """Isolate moving pixels near the projected pelvis, suppressing the moving floor light."""
    image = frame.astype(float)
    x, y = point
    xx, yy = np.meshgrid(np.arange(640), np.arange(360))
    # Limit extraction to the robot's envelope; its center is the recorded pelvis.
    window = np.clip(1.35 - ((xx - x) / 27) ** 4 - ((yy - y) / 52) ** 6, 0, 1)
    x0, x1 = max(0, int(x) - 34), min(640, int(x) + 35)
    y0, y1 = max(0, int(y) - 60), min(360, int(y) + 61)
    roi = np.s_[y0:y1, x0:x1]
    delta = image[roi] - plate[roi]
    # Local illumination changes more slowly than the edges of moving limbs.
    illumination = np.stack([median_filter(channel, size=15) for channel in delta.transpose(2, 0, 1)], axis=2)
    strength = np.max(abs(delta - illumination), axis=2)
    mask = np.zeros((360, 640))
    mask[roi] = np.clip((strength - 5) / 25, 0, 1) * window[roi]
    core = binary_fill_holes(binary_closing(mask > 0.35, iterations=1))
    mask = np.minimum(gaussian_filter(np.maximum(mask, core * 0.92), 0.32), window)
    return np.dstack([frame, np.rint(mask * 255).astype("uint8")])


def figure(reference, dest, qa_scripts):
    summary = json.loads((reference / "result.json").read_text())
    motion = read_rows(reference / "motion.jsonl")
    predictions = read_rows(reference / "predictions.jsonl")
    if (
        summary["outcome"] != "arrived"
        or summary["iteration"] != 3800
        or not summary["wide_nav_camera"]
        or not summary["clearance_guide_m"]
    ):
        raise ValueError("This figure depicts the mapped, dual-camera DA2 3800 arrival.")
    xyz = np.array([summary["start"], *[r["xyz"] for r in motion]])
    times = np.array([r["t"] for r in motion])
    snapshots = [next(r for r in predictions if np.isclose(r["sim_time"], t)) for t in QUERIES]
    reader = imageio.get_reader(reference / "navigation.mp4")
    try:
        # Film writer emits after steps 0, 2, ... and once at termination.
        steps = np.unique(np.r_[np.arange(0, len(motion), 2), len(motion) - 1])
        if reader.count_frames() != len(steps):
            raise ValueError("Unexpected film/physics alignment")

        def frame(t):
            i = np.argmin(abs(times[steps] - t))
            return reader.get_data(int(i)), steps[i]

        first, _ = frame(times[0])
        last, _ = frame(times[-1])
        # These two stationary-camera views supply undisturbed halves of the empty scene.
        plate = first[82:442, :640].copy()
        plate[:, :310] = last[82:442, :310]
        _, project = review_camera(summary["tile"])
        overlays = []
        for t, alpha in [*EXPOSURES, (times[-1], 1)]:
            film, step = frame(t)
            body = get_foreground(film[82:442, :640], plate, project(xyz[step + 1]))
            body[:, :, 3] = np.rint(body[:, :, 3] * alpha)
            overlays.append(body)
        matched = snapshots[0]
        nav_film, nav_step = frame(matched["sim_time"] + 0.02)
        nav = nav_film[82:256, 648:952]
    finally:
        reader.close()
    dest.mkdir(exist_ok=True, parents=True)
    qa = dest / "qa"
    qa.mkdir(exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "text.color": INK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )
    width, height = 180, 131
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), facecolor="white")

    def axes(x, y, w, h):
        ax = fig.add_axes([x / width, (height - y - h) / height, w / width, h / height])
        ax.set_axis_off()
        return ax

    def label(x, y, s, **kw):
        fig.text(x / width, 1 - y / height, s, va="center", **kw)

    def heading(x, y, letter, title):
        label(x, y, letter, fontsize=9, fontweight="bold")
        label(x + 5, y, title, fontsize=8.3, fontweight="bold")

    heading(3, 4, "a", "LightNav-0 and DA2 stair traversal")
    label(177, 4, f"Upper landing  ·  {summary['sim_seconds']:.2f} s", fontsize=7.1, ha="right", color=MUTED)
    label(3, 10, "“" + summary["instruction"] + "”", fontsize=8.15, fontstyle="italic")
    hero = axes(3, 14, 174, 62)
    hero.imshow(
        last[82:442, :640], interpolation="bilinear", extent=(-0.5, 639.5, 359.5, -0.5), aspect="auto"
    )
    xy_img = track_on_terrain(xyz, summary)
    hero.plot(*xy_img.T, color="#e4fffc", lw=2.0, solid_capstyle="round")
    hero.plot(*xy_img.T, color=TRACK, lw=1.15, solid_capstyle="round")
    selected_plan = track_on_terrain(np.array(matched["plan_xy"]), summary)
    hero.plot(*selected_plan.T, color="#ffefd8", lw=2.0, zorder=2.1)
    hero.plot(*selected_plan.T, color=PLAN, lw=0.85, zorder=2.2)
    hero.scatter(*selected_plan[1:].T, s=6.5, fc=PLAN, ec="#fff5e8", lw=0.45, zorder=2.3)
    hero.annotate(
        f"{matched['sim_time']:.2f} s plan",
        xy=selected_plan[5],
        xytext=(201, 268),
        ha="center",
        fontsize=6.2,
        color="#753006",
        arrowprops=dict(arrowstyle="-", lw=0.6, color="#8d491d", shrinkB=2.2),
    )
    # PIL composite preserves the alpha of each earlier exposure and the original film coordinates.
    layer = Image.new("RGBA", (640, 360))
    for body in overlays:
        layer.alpha_composite(Image.fromarray(body))
    hero.imshow(layer, extent=(-0.5, 639.5, 359.5, -0.5), interpolation="bilinear", aspect="auto", zorder=4)
    hero.set_xlim(24, 624)
    hero.set_ylim(284, 284 - 600 * 62 / 174)
    heading(3, 81, "b", "Local plans and executed path")
    heading(112, 81, "c", "Navigation view")
    plan = axes(3, 89, 100, 37.2)
    plan.set_xlim(-0.53, 9.04)
    plan.set_ylim(-1.78, 1.78)
    plan.set_aspect("auto")
    # Keep true, equal metric scales in the floor plan.
    plan.set_facecolor("white")
    stair_edges = summary["tile"]["edges"]
    for i, left in enumerate(stair_edges):
        right = stair_edges[i + 1] if i + 1 < len(stair_edges) else 8.83
        plan.add_patch(Rectangle((left, -2), right - left, 4, fc="#f1f2ef", ec="none", zorder=0))
        plan.plot([left, left], [-1.78, 1.78], color="#bdc8c8", lw=0.58, zorder=1)
    for obj in summary["fixtures"]:
        w, h = obj["length"], obj["width"]
        fc = "#e2cfb1" if obj["style"] == "crate" else "#e1a17e"
        plan.add_patch(
            Rectangle((obj["x"] - w / 2, obj["y"] - h / 2), w, h, fc=fc, ec="#9e896d", lw=0.45, zorder=2)
        )
    plan.plot(*xyz[:, :2].T, lw=1.55, color=TRACK, zorder=3, solid_capstyle="round")
    for snap, label_x in zip(snapshots, [1.4, 4.2, 7.77]):
        path = np.array(snap["plan_xy"])
        plan.plot(*path.T, lw=0.76, color=PLAN, zorder=5)
        plan.scatter(*path[1:].T, s=4.1, fc=PLAN, ec="white", lw=0.2, zorder=6)
        x, y = path[0]
        plan.scatter(x, y, s=14, c="white", ec=PLAN, lw=0.9, zorder=7)
        plan.annotate(
            f"{snap['sim_time']:.2f} s",
            xy=(x, y - 0.07),
            xytext=(label_x, -0.86),
            ha="center",
            va="center",
            fontsize=5.6,
            color=MUTED,
            arrowprops=dict(arrowstyle="-", lw=0.45, color="#73878b", shrinkA=1.7, shrinkB=1.5),
        )
    plan.plot(*xyz[0, :2], marker="o", ms=3.2, mfc=TRACK, mec="white", mew=0.45, zorder=7)
    plan.plot(*summary["goal_xy"], marker="D", ms=4, mfc="white", mec=INK, mew=0.8, zorder=7)
    plan.text(8.1, 0.37, "Goal", ha="center", fontsize=5.8)
    plan.plot([-0.15, 0.85], [-1.14, -1.14], color=INK, lw=0.9)
    plan.text(0.35, -1.0, "1 m", ha="center", fontsize=5.2)
    fig.legend(
        [Line2D([], [], lw=1.55, color=TRACK), Line2D([], [], lw=0.8, color=PLAN, marker="o", ms=2.2)],
        ["Executed path", "LightNav waypoints"],
        loc="center left",
        bbox_to_anchor=(3 / width, 1 - 86 / height),
        frameon=False,
        ncol=2,
        handlelength=2.0,
        columnspacing=1.5,
        borderaxespad=0,
        fontsize=6.4,
    )
    nav_ax = axes(112, 89, 65, 37.2)
    nav_ax.imshow(nav, aspect="auto", interpolation="bilinear")
    label(112, 86, "Wide-angle RGB with projected local plan", fontsize=6.0, color=MUTED)
    label(177, 86, f"{times[nav_step]:.2f} s", fontsize=6, ha="right", color=MUTED)
    label(3, 129, "Dual cameras  ·  Map-assisted tracking  ·  One simulated rollout", fontsize=6, color=MUTED)
    fig.canvas.draw()
    if qa_scripts is not None:
        sys.path.insert(0, str(qa_scripts))
        from audit_panel_alignment import require_matplotlib_panel_alignment

        require_matplotlib_panel_alignment(
            fig,
            json_out=qa / "alignment.json",
            strict=True,
            panel_ids=["scene", "plan", "rgb"],
            row_groups=[["plan", "rgb"]],
            column_groups=[],
        )
    fig.savefig(dest / "lightnav_stairs.pdf", dpi=330, facecolor="white")
    fig.savefig(dest / "lightnav_stairs.svg", dpi=330, facecolor="white")
    fig.savefig(dest / "lightnav_stairs.png", dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--qa-scripts", type=Path, help="Optional directory containing audit_panel_alignment.py")
    args = parser.parse_args()
    figure(args.reference, args.output, args.qa_scripts)
