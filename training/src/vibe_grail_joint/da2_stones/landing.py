"""Training-only landing support and toe/sole clearance at the next pillar edge."""
import copy
import math

import torch


def validate_settings(settings):
    stones = settings['stones']
    minimum = stones['minimum_support_fraction']
    if not isinstance(minimum, (int, float)) or not math.isfinite(minimum) or not 0 < minimum <= 1:
        raise ValueError('Minimum support fraction must lie in (0, 1]')
    cfg = stones.get('swing_clearance')
    weight = stones['reward_weights'].get('swing_clearance')
    if cfg is None:
        if weight is not None:
            raise ValueError('Swing clearance reward requires its phase/geometry settings')
        return None
    keys = {'version', 'margin_m', 'lead_distance_m', 'edge_band_m', 'deficit_scale_m', 'max_cost'}
    if set(cfg) != keys or cfg['version'] != 1:
        raise ValueError('Unknown swing clearance configuration')
    for key in keys - {'version'}:
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'Invalid swing clearance {key}')
    if not .01 <= cfg['margin_m'] <= .10:
        raise ValueError('Swing clearance margin must be between 1 and 10 cm')
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight >= 0:
        raise ValueError('Swing clearance deficit must have a negative reward weight')
    return cfg


def valid_top_support(fraction, surface, minimum):
    return (fraction >= minimum) & (surface >= 0)


def normalize_landing_warm_start(source, target):
    """Permit only support/clearance objective changes; physical terrain stays identical."""
    normalized = copy.deepcopy(target)
    if validate_settings(target) is None:
        return normalized, False
    old, new = source['stones'], target['stones']
    changed = any(old.get(key) != new.get(key)
                  for key in ('minimum_support_fraction', 'swing_clearance'))
    changed |= old['reward_weights'].get('swing_clearance') != new['reward_weights'].get('swing_clearance')
    for key in ('minimum_support_fraction', 'swing_clearance'):
        if key in old:
            normalized['stones'][key] = copy.deepcopy(old[key])
        else:
            normalized['stones'].pop(key, None)
    weights = normalized['stones']['reward_weights']
    if 'swing_clearance' in old['reward_weights']:
        weights['swing_clearance'] = old['reward_weights']['swing_clearance']
    else:
        weights.pop('swing_clearance', None)
    return normalized, bool(changed)


def swing_clearance_cost(sole_points, target, origins, can_swing, active, minimum_support, cfg):
    """Penalize insufficient sole height only during entry across a round front edge.

    target uses PrimitiveHeights' [x,y,hx,hy,top,collision_sides] layout.
    The worst (lowest) sole point includes toes and heel, not only ankle height.
    Release once enough sole samples project onto the target to permit landing.
    """
    center = target[:, :2] + origins[:, :2]
    radius = target[:, 2].clamp_min(1e-6)
    dx = sole_points[..., 0] - center[:, None, None, 0]
    dy = sole_points[..., 1] - center[:, None, None, 1]
    # Match the 64-sided collision footprint used by PrimitiveHeights.
    sides = target[:, 5].clamp_min(3)
    phi = torch.pi / sides[:, None, None]
    delta = torch.remainder(torch.atan2(dy, dx), 2 * phi) - phi
    polygon_radius = radius[:, None, None] * torch.cos(phi) / torch.cos(delta)
    footprint_radius = torch.where(target[:, None, None, 5] > 2, polygon_radius, radius[:, None, None])
    projected = (dx.square() + dy.square() <= footprint_radius.square()).float().mean(-1)
    lateral = dy.abs() <= radius[:, None, None]
    front = center[:, None, None, 0] - (
        radius[:, None, None].square() - dy.square()).clamp_min(0).sqrt()
    edge_distance = sole_points[..., 0] - front
    crossing = lateral & (edge_distance >= -cfg['lead_distance_m']) & (edge_distance <= cfg['edge_band_m'])
    # Increase pressure smoothly as the toe approaches the front edge.
    ramp = ((edge_distance + cfg['lead_distance_m']) / cfg['lead_distance_m']).clamp(0, 1)
    gate = torch.where(crossing, ramp, 0.).amax(-1)
    valid_target = (target[:, 2] > 0) & (target[:, 5] > 0)
    gate *= (can_swing & active[:, None] & valid_target[:, None] & (projected < minimum_support))
    height = target[:, 4] + origins[:, 2] + cfg['margin_m']
    deficit = (height[:, None] - sole_points[..., 2].amin(-1)).clamp_min(0)
    per_foot = (deficit / cfg['deficit_scale_m']).square().clamp_max(cfg['max_cost']) * gate
    return per_foot.sum(-1), torch.where(gate > 0, deficit, 0.), gate, projected
