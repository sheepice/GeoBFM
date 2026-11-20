"""VR source routing with continuous enter/exit and planner fallback, no robot I/O."""

import torch


class TargetRouter:
    def __init__(self, transition_seconds=0.5):
        if transition_seconds <= 0:
            raise ValueError("Positive transition duration required")
        self.duration = transition_seconds
        self.value = None
        self.origin = None
        self.active = None
        self.elapsed = 0.0

    def update(
        self, planner_points, planner_orientations, dt, vr_points=None, vr_orientations=None
    ):
        if dt <= 0:
            raise ValueError("Positive dt required")
        enabled = vr_points is not None
        if enabled and vr_orientations is None:
            raise ValueError("VR position and orientation must be provided together")
        target_p = vr_points if enabled else planner_points
        target_q = vr_orientations if enabled else planner_orientations
        if (
            not torch.isfinite(target_p).all()
            or not torch.isfinite(target_q).all()
            or (target_q.norm(dim=-1) < 1e-6).any()
        ):
            raise ValueError("Invalid three-point targets")
        target_q = target_q / target_q.norm(dim=-1, keepdim=True)
        if self.value is None:
            self.value = (planner_points.clone(), planner_orientations.clone())
        if self.active != enabled:
            self.origin = tuple(x.clone() for x in self.value)
            self.active = enabled
            self.elapsed = 0.0
        self.elapsed = min(self.duration, self.elapsed + dt)
        w = self.elapsed / self.duration
        a, b = self.origin
        # Sign-aligned normalized quaternion interpolation avoids the q/-q discontinuity.
        target_q = torch.where((b * target_q).sum(-1, keepdim=True) < 0, -target_q, target_q)
        q = b * (1 - w) + target_q * w
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.value = (a * (1 - w) + target_p * w, q)
        return self.value


class TeleopInputBuilder:
    """Convert planner reference plus optional calibrated VR points into release TELEOP input.

    No VR hardware or network dependency. The caller supplies L/R/head positions
    and wxyz orientations in the full reference-root frame, exactly as training.
    """

    def __init__(self, transition_seconds=0.5, device="cpu"):
        from .fk import RobotFK

        self.fk = RobotFK(device)
        self.router = TargetRouter(transition_seconds)

    def build(
        self,
        lower_position,
        lower_velocity,
        planner_current_joints,
        reference_root_quaternion,
        robot_root_quaternion,
        dt,
        vr_positions=None,
        vr_orientations=None,
    ):
        from .contracts import points_from_bodies, teleop_input, mul, inv

        body_p, body_q = self.fk(planner_current_joints)
        root_p = body_p.new_zeros(*body_p.shape[:-2], 3)
        identity = body_q.new_zeros(*body_q.shape[:-2], 4)
        identity[..., 0] = 1
        planner_p, planner_q = points_from_bodies(
            body_p, body_q, self.fk.body_names, root_p, identity
        )
        p, q = self.router.update(planner_p, planner_q, dt, vr_positions, vr_orientations)
        relative = mul(inv(robot_root_quaternion), reference_root_quaternion)
        return teleop_input(lower_position, lower_velocity, p, q, relative)
