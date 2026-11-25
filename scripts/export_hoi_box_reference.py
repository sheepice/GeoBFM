"""Export a G1 OMOMO reference and check the palms against the scaled collision hull."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def export(clip, output):
    dataset = clip.parent.parent
    manifest = json.loads((dataset / "manifest.json").read_text())
    metadata = json.loads((dataset / "accepted_metadata.json").read_text())
    entry = next(row for row in metadata["accepted"] if row["sequence"] == clip.stem)
    xml = dataset / manifest["mjcf"]
    spec = mujoco.MjSpec.from_file(str(xml))
    visual = spec.mesh("object_mesh")
    scale = entry["effective_mesh_scale"]
    # The OBJ uses an XML scale. The representative hull stores pre-scaled vertices.
    source_scale = visual.scale.copy()
    spec.mesh("object_collision_mesh").scale[:] *= scale / source_scale
    visual.scale[:] = scale
    model = spec.compile()
    data = mujoco.MjData(model)
    with np.load(clip, allow_pickle=False) as motion:
        qpos, fps = motion["qpos"], int(motion["fps"].item())
    split = int(model.joint("object_root").qposadr.item())
    if qpos.shape[1] != model.nq or split != 36 or model.nq != 43:
        raise ValueError("Expected the G1 29-joint model followed by a free box.")
    box = model.body("object").id
    geom = model.geom("object_collision").id
    mesh = model.geom_dataid[geom]
    addr = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[addr : addr + model.mesh_vertnum[mesh]]
    sides = ["left", "right"]
    sites = [model.site(f"{s}_palm").id for s in sides]
    hands = [model.geom(f"{s}_hand_collision").id for s in sides]
    palms = np.empty((len(qpos), 2, 3))
    gaps = np.empty((len(qpos), 2))
    bottom = np.empty(len(qpos))
    for i, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        palms[i] = (data.site_xpos[sites] - data.xpos[box]) @ data.xmat[box].reshape(3, 3)
        for j, hand in enumerate(hands):
            gaps[i, j] = mujoco.mj_geomDistance(model, data, hand, geom, 5.0, None)
        bottom[i] = data.geom_xpos[geom, 2] + (vertices @ data.geom_xmat[geom].reshape(3, 3)[2]).min()

    xyz = qpos[:, split : split + 3]
    start, end = (np.median(window, axis=0) for window in (xyz[:5], xyz[-5:]))
    raised = xyz[:, 2] > max(start[2], end[2]) + 0.12
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "hoi_reference.npz",
        fps=fps,
        time_s=np.arange(len(qpos)) / fps,
        robot_qpos=qpos[:, :split],
        robot_joint_names=manifest["qpos_names"][7:split],
        box_pose_world_wxyz=qpos[:, split:],
        sides=sides,
        palm_pos_in_box_m=palms,
        palm_signed_gap_m=gaps,
        box_bottom_world_z_m=bottom,
    )
    report = {
        "sequence": clip.stem,
        "backend": entry["backend"],
        "motion_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
        "mjcf": xml.name,
        "mjcf_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "reference_type": "kinematics only",
        "frames": len(qpos),
        "fps": fps,
        "elapsed_s": (len(qpos) - 1) / fps,
        "metadata_box_dimensions_m": entry["object_dimensions_m"],
        "mjcf_mass_kg": float(model.body_mass[box]),
        "representative_mesh_scale": source_scale.tolist(),
        "applied_sequence_mesh_scale": scale,
        "xy_displacement_m_median5": float(np.linalg.norm(end[:2] - start[:2])),
        "max_box_origin_z_gain_m_from_start_median5": float(xyz[:, 2].max() - start[2]),
        "bottom_z_m_first_last": bottom[[0, -1]].tolist(),
        "raised_frames_above_endpoint_medians_plus_12cm": int(raised.sum()),
        "palm_signed_gap_m_when_raised": {
            side: dict(
                zip(("p05", "median", "p95"), np.quantile(gaps[raised, j], [0.05, 0.5, 0.95]).tolist())
            )
            for j, side in enumerate(sides)
            if raised.any()
        },
    }
    (output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.clip, args.output)
