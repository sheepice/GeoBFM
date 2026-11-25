"""Estimate the grounded box center from the same DA2 tokens used by the actor."""

import torch
import torch.nn.functional as F
from vibe_grail_joint.da2_locomotion.observation import FrozenVisualTokens


def image_features(tokens):
    # Retain coarse spatial layout while reducing the calibrated regression to 1536 inputs.
    grid = tokens.float().reshape(-1, 8, 8, 96).permute(0, 3, 1, 2)
    return F.avg_pool2d(grid, 2).flatten(1)


def ground_intersection(ray_xy, rotation, origin, plane_z):
    ray = torch.cat([ray_xy, -torch.ones_like(ray_xy[:, :1])], dim=1)
    direction = (rotation @ ray.unsqueeze(-1)).squeeze(-1)
    distance = (plane_z - origin[:, 2]) / direction[:, 2].clamp_max(-0.05)
    return origin[:, :2] + direction[:, :2] * distance[:, None]


def predict_ray(features, calibration):
    x = (features - calibration["mean"]) / calibration["std"]
    x = F.silu(F.linear(x, calibration["hidden"], calibration["hidden_bias"]))
    return F.linear(x, calibration["output"], calibration["output_bias"]) * 0.2


def pickup_xy(tokens, rotation, origin, plane_z, calibration):
    features = image_features(tokens)
    ray_xy = predict_ray(features, calibration)
    return ground_intersection(ray_xy, rotation, origin, plane_z)


class PickupVisualTokens(FrozenVisualTokens):
    """Refresh the target before computing the actor's motion and grasp observations."""

    @torch.no_grad()
    def __call__(self, env, **kwargs):
        tokens = super().__call__(env, **kwargs)
        c = env.command_manager.get_term("motion")
        due = c.full & (c.time_steps <= 10) & (self.history.last_step == int(env.common_step_counter))
        if c.cfg.pickup_input == "black_rgb":
            return tokens  # Missing pickup image uses the nominal reference.
        if c.cfg.pickup_input == "freeze":
            due &= c.time_steps == 0
        ids = (due & (c.last_pickup_tick != c.time_steps)).nonzero().flatten()
        if len(ids):
            cam = env.scene["head_cam"].camera_idx
            for chunk in ids.split(self.settings["encoder_batch_size"]):
                nominal = c.box_pose[0, :3] + env.scene.env_origins[chunk]
                xy = pickup_xy(
                    self.history.values[chunk, -1],
                    env.sim.data.cam_xmat[chunk, cam].reshape(-1, 3, 3),
                    env.sim.data.cam_xpos[chunk, cam],
                    nominal[:, 2],
                    c.calibration,
                )
                delta = (xy - nominal[:, :2]).clamp(-0.36, 0.36)
                prior = c.pickup_offset[chunk, :2]
                # Three independent frames while the object is stationary, before hand contact.
                weight = torch.where(c.last_pickup_tick[chunk] < 0, 1.0, 0.35)
                c.pickup_offset[chunk, :2] = torch.lerp(prior, delta, weight[:, None])
                c.last_pickup_tick[chunk] = c.time_steps[chunk]
        return tokens
