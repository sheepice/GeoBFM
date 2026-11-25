"""Reference-based upper-body PD feedforward for the free, two-hand carry."""

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_apply
from scipy.ndimage import gaussian_filter1d


class GraspFeedback:
    def __init__(self, env, cmd):
        model = env.sim._mj_model
        data = mujoco.MjData(model)
        self.arm = [
            [
                i
                for i, name in enumerate(cmd.robot.joint_names)
                if name.startswith(side + "_") and any(k in name for k in ("shoulder", "elbow", "wrist"))
            ]
            for side in ("left", "right")
        ]
        assert [len(g) for g in self.arm] == [7, 7]
        all_joints = [model.joint("robot/" + n).id for n in cmd.robot.joint_names]
        motor = [np.flatnonzero(model.actuator_trnid[:, 0] == j).item() for j in all_joints]
        kp = model.actuator_gainprm[motor, 0]
        assert np.all(kp > 0)
        sites = [model.site("robot/" + s + "_palm").id for s in ("left", "right")]
        torso = model.body("robot/torso_link").id
        box = model.body("box/object").id
        with np.load(cmd.cfg.motion_file) as ref:
            poses = np.column_stack((ref["qpos"], ref["box_pose"]))
            grip = gaussian_filter1d(ref["grip"].astype(float), 2, axis=0)
        inv_jac = np.zeros((len(poses), 2, 7, 3))
        palm_relative = np.zeros((len(poses), 2, 3))
        offset = np.zeros((len(poses), len(all_joints)))
        for t, q in enumerate(poses):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            palms = data.site_xpos[sites]
            palm_relative[t] = (palms - data.xpos[box]) @ data.xmat[box].reshape(3, 3)
            inward = palms[::-1] - palms
            inward /= np.maximum(np.linalg.norm(inward, axis=-1, keepdims=True), 1e-6)
            torque = data.qfrc_bias[model.jnt_dofadr[all_joints]].copy()
            for hand, joints in enumerate(self.arm):
                jac = np.empty((3, model.nv))
                mujoco.mj_jacSite(model, data, jac, None, sites[hand])
                cols = model.jnt_dofadr[np.array(all_joints)[joints]]
                J = jac[:, cols]
                local = data.xmat[torso].reshape(3, 3).T @ J
                inv_jac[t, hand] = local.T @ np.linalg.inv(local @ local.T + 0.045**2 * np.eye(3))
                # Each hand supplies half the box weight and an opposed normal load.
                force = grip[t, hand] * (28.0 * inward[hand] + [0.0, 0.0, model.body_mass[box] * 9.81 / 2])
                torque[joints] += J.T @ force
            offset[t] = np.clip(torque / kp, -1.15, 1.15)
        to_device = lambda x: torch.as_tensor(x, device=env.device, dtype=torch.float32)
        self.jac = to_device(inv_jac)
        self.palms = to_device(palm_relative)
        self.bias = to_device(offset)
        self.grip = to_device(grip)
        self.torso_index = cmd.robot.body_names.index("torso_link")

    def target(self, cmd):
        t = cmd.time_steps
        future = (t + 2).clamp(max=cmd.motion.time_step_total - 1)
        base = cmd.motion.joint_pos[future] + self.bias[t]
        box = cmd.box.data
        target_box = cmd.box_target(future)
        observed = cmd.observed_box_position()
        correction = (target_box - observed).clamp(-0.09, 0.09)
        center = observed + 0.55 * correction
        quat = box.root_link_quat_w[:, None].expand(-1, 2, -1)
        palms = center[:, None] + quat_apply(quat.reshape(-1, 4), self.palms[t].reshape(-1, 3)).view(-1, 2, 3)
        hand_error = palms - cmd.robot.data.site_pos_w[:, cmd.sites]
        torso = cmd.robot.data.body_link_quat_w[:, self.torso_index, None].expand(-1, 2, -1)
        local = quat_apply_inverse(torso.reshape(-1, 4), hand_error.reshape(-1, 3)).view(-1, 2, 3)
        dq = (self.jac[t] @ local[:, :, :, None]).squeeze(-1).clamp(-0.55, 0.55)
        for hand, joints in enumerate(self.arm):
            ik = cmd.robot_joint_pos[:, joints] + dq[:, hand] - cmd.motion.joint_pos[future][:, joints]
            base[:, joints] += 0.75 * self.grip[t, hand, None] * ik.clamp(-0.55, 0.55)
        return base
