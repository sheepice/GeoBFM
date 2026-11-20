"""Batch root-local G1 FK, verified against MuJoCo; never queries terrain."""

import numpy as np
import torch
from .contracts import mul, rotate


class RobotFK:
    def __init__(self, device="cpu"):
        from vibe_grail_joint.prepare import build_model
        import mujoco

        self.model, self.body_names = build_model()
        m = self.model
        self.root = next(
            i
            for i in range(1, m.nbody)
            if any(
                m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                for j in range(m.body_jntadr[i], m.body_jntadr[i] + m.body_jntnum[i])
            )
        )
        self.joint_names = [
            m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        self.tracked = [m.body(n).id for n in self.body_names]
        self.parent = m.body_parentid.tolist()
        self.pos = torch.as_tensor(m.body_pos, dtype=torch.float32, device=device)
        self.quat = torch.as_tensor(m.body_quat, dtype=torch.float32, device=device)
        self.joints = {}
        slot = 0
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                body = int(m.jnt_bodyid[j])
                if body in self.joints:
                    raise ValueError("FK assumes one hinge per G1 body")
                self.joints[body] = (
                    slot,
                    torch.as_tensor(m.jnt_axis[j], dtype=torch.float32, device=device),
                    torch.as_tensor(m.jnt_pos[j], dtype=torch.float32, device=device),
                    float(m.qpos0[m.jnt_qposadr[j]]),
                )
                slot += 1
        self.limits = torch.as_tensor(
            np.array([m.jnt_range[m.joint(n).id] for n in self.joint_names]),
            dtype=torch.float32,
            device=device,
        )

    def __call__(self, joints):
        shape = joints.shape[:-1]
        zero = joints.new_zeros(*shape, 3)
        ident = joints.new_zeros(*shape, 4)
        ident[..., 0] = 1
        positions = [zero]
        quaternions = [ident]
        for i in range(1, self.model.nbody):
            if i == self.root:
                positions.append(zero)
                quaternions.append(ident)
                continue
            local_p = self.pos[i].expand(*shape, 3)
            local_q = self.quat[i].expand(*shape, 4)
            if i in self.joints:
                slot, axis, pivot, reference = self.joints[i]
                angle = (joints[..., slot] - reference) / 2
                turn = torch.cat((angle.cos()[..., None], angle.sin()[..., None] * axis), -1)
                turned = mul(local_q, turn)
                local_p = (
                    local_p
                    + rotate(local_q, pivot.expand(*shape, 3))
                    - rotate(turned, pivot.expand(*shape, 3))
                )
                local_q = turned
            parent = self.parent[i]
            positions.append(positions[parent] + rotate(quaternions[parent], local_p))
            quaternions.append(mul(quaternions[parent], local_q))
        return torch.stack([positions[i] for i in self.tracked], -2), torch.stack(
            [quaternions[i] for i in self.tracked], -2
        )
