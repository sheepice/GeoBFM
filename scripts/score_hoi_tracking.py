"""Score frame-synchronized whole-body tracking on recorded carries."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML


class UpperBodyTracking:
    def __init__(self, reference):
        self.model = mujoco.MjModel.from_xml_path(str(G1_XML))
        names = [self.model.joint(i).name for i in range(1, self.model.njnt)]
        with np.load(reference) as ref:
            assert names == list(ref["joint_names"])
            self.qref = ref["qpos"]
            self.release = np.flatnonzero(ref["grip"].any(axis=1))[-1] + 1
        self.arm = [7 + i for i, n in enumerate(names) if any(k in n for k in ("shoulder", "elbow", "wrist"))]
        self.waist = [7 + i for i, n in enumerate(names) if "waist" in n]
        self.legs = [7 + i for i, n in enumerate(names) if any(k in n for k in ("hip", "knee", "ankle"))]
        self.knees = [7 + names.index(f"{s}_knee_joint") for s in ("left", "right")]
        self.ankles = [self.model.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]
        assert len(self.arm) == 14 and len(self.waist) == 3
        self.torso = self.model.body("torso_link").id
        self.limbs = [
            self.model.body(s + "_" + k + "_link").id
            for s in ("left", "right")
            for k in ("elbow", "wrist_yaw")
        ]
        self.ref_quat, self.ref_limbs, self.ref_ankles = self.body_poses(self.qref)

    def body_poses(self, sequence):
        data = mujoco.MjData(self.model)
        quat, limbs, ankles = [], [], []
        for pose in sequence:
            data.qpos[:] = pose
            mujoco.mj_kinematics(self.model, data)
            quat.append(data.xquat[self.torso].copy())
            ankles.append(data.xpos[self.ankles].copy())
            limbs.append(
                (data.xpos[self.limbs] - data.xpos[self.torso]) @ data.xmat[self.torso].reshape(3, 3)
            )
        return np.array(quat), np.array(limbs), np.array(ankles)

    def measure(self, recorded):
        q = recorded[:, : self.model.nq]
        # The initial pose is frame 0. Reward and contact instrumentation see
        # the command used for the physics step before the next command update.
        t = np.clip(np.arange(len(q)) - 1, 0, len(self.qref) - 1)
        target = self.qref[t]
        assert np.max(np.abs(q[0] - target[0])) < 0.025
        angles, limbs, ankles = self.body_poses(q)
        arm_error = np.abs(np.rad2deg(q[:, self.arm] - target[:, self.arm]))
        dot = np.abs((angles * self.ref_quat[t]).sum(-1)).clip(0, 1)
        released = t >= self.release
        return dict(
            frames=len(q),
            reference_frames=len(self.qref),
            arm_mae_deg=float(arm_error.mean()),
            leg_mae_deg=float(np.abs(np.rad2deg(q[:, self.legs] - target[:, self.legs])).mean()),
            knee_mae_deg_left_right=np.abs(np.rad2deg(q[:, self.knees] - target[:, self.knees]))
            .mean(0)
            .tolist(),
            ankle_world_mae_cm_left_right=(
                100 * np.linalg.norm(ankles - self.ref_ankles[t], axis=-1).mean(0)
            ).tolist(),
            pelvis_height_mae_cm=float(100 * np.abs(q[:, 2] - target[:, 2]).mean()),
            waist_mae_deg=float(np.abs(np.rad2deg(q[:, self.waist] - target[:, self.waist])).mean()),
            torso_world_angle_deg=float(np.degrees(2 * np.arccos(dot)).mean()),
            elbow_wrist_torso_mae_cm=float(100 * np.linalg.norm(limbs - self.ref_limbs[t], axis=-1).mean()),
            after_release_arm_mae_deg=float(arm_error[released].mean()) if released.any() else None,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-arm-mae-deg", type=float)
    args = parser.parse_args()
    with np.load(args.replay) as trial:
        result = UpperBodyTracking(args.reference).measure(trial["qpos"])
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.write_text(report + "\n")
    if args.max_arm_mae_deg is not None and result["arm_mae_deg"] > args.max_arm_mae_deg:
        raise SystemExit(1)
