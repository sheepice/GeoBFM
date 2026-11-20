"""Exact SONIC release TELEOP layout, quaternion math and three-point convention."""

import torch

POINT_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link")
POINT_OFFSETS = ((0.18, -0.025, 0.0), (0.18, 0.025, 0.0), (0.0, 0.0, 0.35))
TELEOP_DIM = 267  # 10*12 positions + 10*12 velocities + 3*3 positions + 3*4 orientations + 6 root orientation


def mul(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat(
        (
            aw * bw - (av * bv).sum(-1, keepdim=True),
            aw * bv + bw * av + torch.cross(av, bv, dim=-1),
        ),
        -1,
    )


def inv(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1) / q.square().sum(-1, keepdim=True).clamp_min(
        1e-12
    )


def rotate(q, v):
    q = q.expand(*v.shape[:-1], 4)
    uv = torch.cross(q[..., 1:], v, dim=-1)
    return v + 2 * (q[..., :1] * uv + torch.cross(q[..., 1:], uv, dim=-1))


def rotation6(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
        ),
        -1,
    )


def heading(q):
    w, x, y, z = q.unbind(-1)
    angle = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) / 2
    zero = torch.zeros_like(angle)
    return torch.stack((angle.cos(), zero, zero, angle.sin()), -1)


def points_from_bodies(body_pos, body_quat, body_names, root_pos, root_quat):
    ids = [list(body_names).index(n) for n in POINT_BODIES]
    q = body_quat[..., ids, :]
    offsets = body_pos.new_tensor(POINT_OFFSETS).expand(*q.shape[:-1], 3)
    world = body_pos[..., ids, :] + rotate(q, offsets)
    local = rotate(inv(root_quat)[..., None, :], world - root_pos[..., None, :])
    # Match the release training observation: positions AND orientations use full inverse root.
    orientation = mul(inv(root_quat)[..., None, :], q)
    return local, orientation


def teleop_input(lower_pos, lower_vel, points, orientations, root_relative_quat):
    if lower_pos.shape[-2:] != (10, 12) or lower_vel.shape != lower_pos.shape:
        raise ValueError("TELEOP requires ten lower-body reference frames in MuJoCo leg order")
    return torch.cat(
        (
            lower_pos.flatten(-2),
            lower_vel.flatten(-2),
            points.flatten(-2),
            orientations.flatten(-2),
            rotation6(root_relative_quat),
        ),
        -1,
    )


def physical_action_loss(
    student, teacher, scales, leg_weight=1.0, waist_weight=1.0, arm_weight=1.0
):
    delta = (student - teacher) * scales
    losses = torch.stack(
        (
            delta[..., 0:12].square().mean(),
            delta[..., 12:15].square().mean(),
            delta[..., 15:29].square().mean(),
        )
    )
    weights = delta.new_tensor((leg_weight, waist_weight, arm_weight))
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8), losses
