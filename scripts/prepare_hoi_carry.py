"""Fit OMOMO support and bimanual targets, then export a 50 Hz SONIC motion."""

import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


def rotation(quat):
    return Rotation.from_quat(quat, scalar_first=True)


def resample_quat(frames, query, values):
    return Slerp(frames, rotation(values))(query).as_quat(scalar_first=True)


def angular_velocity(quat):
    size = quat.shape
    delta = (rotation(quat[1:].reshape(-1, 4)) * rotation(quat[:-1].reshape(-1, 4)).inv()).as_rotvec()
    delta = delta.reshape((size[0] - 1, *size[1:-1], 3)) * 50
    return np.concatenate([delta[:1], (delta[:-1] + delta[1:]) * 0.5, delta[-1:]])


def prepare(clip, output):
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

    dataset = clip.parent.parent
    manifest = json.loads((dataset / "manifest.json").read_text())
    entry = next(
        r
        for r in json.loads((dataset / "accepted_metadata.json").read_text())["accepted"]
        if r["sequence"] == clip.stem
    )
    output.mkdir(parents=True, exist_ok=True)
    source = ET.parse(dataset / manifest["mjcf"])
    obj_xml = ET.Element("mujoco", model="omomo_box")
    asset = ET.SubElement(obj_xml, "asset")
    visual = copy.deepcopy(source.find(".//mesh[@name='object_mesh']"))
    hull = copy.deepcopy(source.find(".//mesh[@name='object_collision_mesh']"))
    ratio = entry["effective_mesh_scale"] / float(visual.get("scale").split()[0])
    visual.set("scale", " ".join([str(entry["effective_mesh_scale"])] * 3))
    visual.set("file", str((dataset / "meshes" / visual.get("file")).resolve()))
    hull.set("scale", " ".join([str(ratio)] * 3))
    asset.extend([visual, hull])
    body = copy.deepcopy(source.find(".//body[@name='object']"))
    body.find("geom[@name='object_visual']").set("rgba", "0.71 0.42 0.22 1")
    body.find("geom[@name='object_visual']").set("group", "2")
    body.find("geom[@name='object_collision']").set("group", "3")
    ET.SubElement(obj_xml, "worldbody").append(body)
    box_file = output / "box.xml"
    ET.ElementTree(obj_xml).write(box_file, encoding="unicode")
    spec = mj.MjSpec.from_file(str(box_file))
    spec.worldbody.add_geom(name="floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1])
    table = spec.compile()
    d = mj.MjData(table)
    with np.load(clip) as a:
        q = a["qpos"].copy()
        fps = int(a["fps"])
    original = q.copy()
    spec = mj.MjSpec.from_file(str(dataset / manifest["mjcf"]))
    spec.mesh("object_collision_mesh").scale[:] *= ratio
    spec.mesh("object_mesh").scale[:] = entry["effective_mesh_scale"]
    geom_model = spec.compile()
    fk = mj.MjData(geom_model)
    ground, rest = [], []
    for pose in [q[0, 36:], q[-1, 36:]]:
        mj.mj_resetData(table, d)
        d.qpos[:] = pose
        for _ in range(3500):
            mj.mj_step(table, d)
        rest.append(d.qpos.copy())
        ground.append(float(np.linalg.norm(d.qvel)))
    if max(ground) > 0.005:
        raise RuntimeError(f"Box failed to settle: {ground}")
    feet = [
        i
        for i in range(geom_model.ngeom)
        if "_foot" in geom_model.geom(i).name and "collision" in geom_model.geom(i).name
    ]
    lows = []
    for pose in q:
        fk.qpos[:] = pose
        mj.mj_kinematics(geom_model, fk)
        lows.append(
            np.min(
                fk.geom_xpos[feet, 2]
                - geom_model.geom_size[feet, 0]
                - np.abs(fk.geom_xmat[feet, 8]) * geom_model.geom_size[feet, 1]
            )
        )
    q[:, 2] -= gaussian_filter1d(lows, 1.5)
    for endpoint, resting, which in [
        (0, rest[0], np.arange(len(q)) < len(q) // 2),
        (-1, rest[1], np.arange(len(q)) >= len(q) // 2),
    ]:
        weight = np.clip(1 - (original[:, 38] - original[endpoint, 38]) / 0.23, 0, 1)
        weight = weight**2 * (3 - 2 * weight) * which
        q[:, 38] += weight * (resting[2] - original[endpoint, 38])
        correction = rotation(resting[3:]) * rotation(original[endpoint, 39:]).inv()
        q[:, 39:] = (
            Rotation.from_rotvec(weight[:, None] * correction.as_rotvec()) * rotation(q[:, 39:])
        ).as_quat(scalar_first=True)
    obj = geom_model.geom("object_collision").id
    box = geom_model.body("object").id
    desired = []
    before, after = [], []
    for side in ["left", "right"]:
        hand = geom_model.geom(f"{side}_hand_collision").id
        site = geom_model.site(f"{side}_palm").id
        idx = np.array(
            [
                j
                for j in range(1, geom_model.njnt - 1)
                if geom_model.joint(j).name.startswith(side + "_")
                and any(k in geom_model.joint(j).name for k in ("shoulder", "elbow", "wrist"))
            ]
        )
        cols = geom_model.jnt_qposadr[idx]
        low, high = geom_model.jnt_range[idx].T
        gaps, local, initial = [], [], []
        for pose in original:
            fk.qpos[:] = pose
            mj.mj_kinematics(geom_model, fk)
            initial.append(mj.mj_geomDistance(geom_model, fk, hand, obj, 3, None))
            local.append((fk.site_xpos[site] - fk.xpos[box]) @ fk.xmat[box].reshape(3, 3))
        local = np.asarray(local)
        # Continuous intended grip from reaching the sides until controlled release.
        near = np.flatnonzero((np.array(initial) < 0.035) & (np.arange(len(q)) < len(q) // 2))
        late = np.flatnonzero((np.array(initial) < 0.035) & (np.arange(len(q)) >= len(q) // 2))
        if not (near.size and late.size):
            raise ValueError(f"Could not locate {side} grip and release")
        grip = (np.arange(len(q)) >= near[0]) & (np.arange(len(q)) <= late[-1])
        desired.append(grip)
        prior = np.zeros(7)
        for i, pose in enumerate(q):
            base = original[i, cols]
            target = pose[36:39] + rotation(pose[39:]).apply(local[i])

            def residual(angles):
                fk.qpos[:] = pose
                fk.qpos[cols] = angles
                mj.mj_kinematics(geom_model, fk)
                gap = mj.mj_geomDistance(geom_model, fk, hand, obj, 3, None)
                err = gap + 0.001 if grip[i] else min(gap, 0)
                return np.r_[
                    8 * err,
                    (fk.site_xpos[site] - target) * (1 if grip[i] else 0.18),
                    0.055 * (angles - base),
                    0.045 * (angles - base - prior),
                ]

            fit = least_squares(residual, np.clip(base + prior, low, high), bounds=(low, high), max_nfev=28)
            pose[cols] = fit.x
            prior = fit.x - base
            residual(fit.x)
            gaps.append(mj.mj_geomDistance(geom_model, fk, hand, obj, 3, None))
        before.append(np.quantile(np.array(initial)[grip], [0.05, 0.5, 0.95]).tolist())
        after.append(np.quantile(np.array(gaps)[grip], [0.05, 0.5, 0.95]).tolist())
    desired = np.array(desired).T
    tick = np.arange(round((len(q) - 1) / fps * 50) + 1 + 30)
    frames = np.arange(len(q))
    query = np.minimum(tick * fps / 50, len(q) - 1)
    interpolated = PchipInterpolator(frames, q, axis=0)(query)
    for begin in [3, 39]:
        interpolated[:, begin : begin + 4] = resample_quat(frames, query, q[:, begin : begin + 4])
    robot = mj.MjModel.from_xml_path(str(G1_XML))
    rd = mj.MjData(robot)
    names = [robot.joint(i).name for i in range(1, robot.njnt)]
    order = [manifest["qpos_names"].index(n) for n in names]
    rq = interpolated[:, [0, 1, 2, 3, 4, 5, 6, *order]]
    pos, quat, palms = [], [], []
    for rpose, pose in zip(rq, interpolated, strict=True):
        rd.qpos[:] = rpose
        mj.mj_kinematics(robot, rd)
        pos.append(rd.xpos[1:].copy())
        quat.append(rd.xquat[1:].copy())
        fk.qpos[:] = pose
        mj.mj_kinematics(geom_model, fk)
        sites = [geom_model.site(side + "_palm").id for side in ("left", "right")]
        palms.append((fk.site_xpos[sites] - fk.xpos[box]) @ fk.xmat[box].reshape(3, 3))
    pos, quat = np.array(pos), np.array(quat)
    bp = interpolated[:, 36:43]
    # Project collision vertices into the box free-joint frame once for support-height metrics.
    d.qpos[:] = [0, 0, 0, 1, 0, 0, 0]
    mj.mj_forward(table, d)
    g = table.geom("object_collision").id
    mesh = table.geom_dataid[g]
    addr = table.mesh_vertadr[mesh]
    verts = (
        table.mesh_vert[addr : addr + table.mesh_vertnum[mesh]] @ d.geom_xmat[g].reshape(3, 3).T
        + d.geom_xpos[g]
    )
    np.savez_compressed(
        output / "motion.npz",
        fps=50,
        source_frame=query,
        qpos=rq,
        joint_pos=rq[:, 7:],
        joint_vel=np.gradient(rq[:, 7:], 0.02, axis=0),
        body_pos_w=pos,
        body_quat_w=quat,
        body_lin_vel_w=np.gradient(pos, 0.02, axis=0),
        body_ang_vel_w=angular_velocity(quat),
        joint_names=names,
        body_names=[robot.body(i).name for i in range(1, robot.nbody)],
        box_pose=bp,
        box_lin_vel=np.gradient(bp[:, :3], 0.02, axis=0),
        box_ang_vel=angular_velocity(bp[:, 3:]),
        palms=np.array(palms),
        grip=desired[np.rint(query).astype(int)],
        vertices=verts,
    )
    report = dict(
        sequence=clip.stem,
        source_fps=fps,
        control_hz=50,
        total_ticks=len(tick),
        hold_ticks=30,
        box_mass_kg=float(table.body_mass[table.body("object").id]),
        rest_speed_norm=ground,
        reference_grip_signed_gap_m_p05_median_p95_before=before,
        reference_grip_signed_gap_m_p05_median_p95_after=after,
        settled_box_start_end=[x.tolist() for x in rest],
    )
    (output / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("clip", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    prepare(a.clip, a.output)
