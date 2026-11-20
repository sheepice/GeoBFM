"""Yellow-course layouts and straight crossing routes for the final curriculum."""

import copy
import math

import numpy as np
import torch

from .layout_randomization import refresh


LEVEL = 10
LONGITUDINAL, PERPENDICULAR, OBLIQUE = range(3)


def route_groups(env_ids):
    """Assign one fourth of the pillar slots to lengthwise approaches, the remainder equally."""
    ordinal = env_ids.div(20, rounding_mode='floor') * 12 + env_ids.remainder(20)
    slot = ordinal.remainder(8)
    longitudinal = slot.remainder(4) == 0
    perpendicular = (slot == 1) | (slot == 3) | (slot == 6)
    return torch.where(longitudinal, LONGITUDINAL, torch.where(perpendicular, PERPENDICULAR, OBLIQUE))


def _heights(nominal, rng, jitter, maximum_difference):
    # Draw every pillar independently, then condition on the approved step-height limit.
    for _ in range(10000):
        heights = nominal + rng.uniform(-jitter, jitter, len(nominal))
        if heights.min() > 0 and np.max(np.abs(np.diff(heights))) <= maximum_difference:
            return heights
    raise ValueError('Unable to sample positive pillar heights within the step-height limit')


def _with_heights(tile, heights):
    tile = copy.deepcopy(tile)
    for stone, height in zip(tile['solids'], heights, strict=True):
        stone['height_above_mat_m'] = float(height)
        stone['top'] = stone['bottom'] + float(height)
    refresh(tile)
    return tile


def _crossing(tile, rng, kind, sign):
    centers = np.array([(s['x'], s['y']) for s in tile['solids']])
    radii = np.array([s['hx'] for s in tile['solids']])
    mats = tile['mats']
    xmin, xmax = mats[0]['x'] - mats[0]['hx'], mats[-1]['x'] + mats[-1]['hx']
    for _ in range(64):
        if kind == PERPENDICULAR:
            first = int(rng.integers(len(centers)))
            targets = [first]
            axis = np.array([0., sign])
        else:
            first = int(rng.integers(len(centers) - 1))
            targets = [first, first + 1]
            if (centers[targets[1], 1] - centers[targets[0], 1]) * sign < 0:
                targets.reverse()
            axis = centers[targets[1]] - centers[targets[0]]
            axis /= np.linalg.norm(axis)
            angle = math.atan2(axis[1], axis[0]) + rng.uniform(-.035, .035)
            axis = np.array([math.cos(angle), math.sin(angle)])
        anchor = centers[targets[0]]
        entry = (-sign * .5 - anchor[1]) / axis[1]
        exit = (sign * .5 - anchor[1]) / axis[1]
        crossing_x = anchor[0] + np.array([entry, exit]) * axis[0]
        # Leave room for the feet on both sides of the crossing line near the mat ends.
        if np.any(crossing_x <= xmin + .13) or np.any(crossing_x >= xmax - .13):
            continue
        lateral = (centers - anchor) @ np.array([-axis[1], axis[0]])
        if np.any(np.abs(lateral[targets]) > radii[targets] - .095):
            continue
        others = np.ones(len(centers), dtype=bool)
        others[targets] = False
        # Keep the 18 cm lateral landing corridor clear of non-target tops.
        if np.any(np.abs(lateral[others]) < radii[others] + .09):
            continue
        if len(targets) == 2 and not .45 <= np.linalg.norm(np.diff(centers[targets], axis=0)) <= .82:
            continue
        return anchor, axis, targets, float(entry), float(exit)
    raise ValueError('No usable crossing route in this layout')


def _orient(tile, rng, kind, sign):
    if kind == LONGITUDINAL:
        tile.update(route_kind=kind, route_targets=list(range(len(tile['solids']))),
                    route_first_foot=0 if tile['solids'][0]['y'] > 0 else 1,
                    route_exit=tile['course_end'])
        return tile

    anchor, axis, targets, entry, exit = _crossing(tile, rng, kind, sign)
    left = np.array([-axis[1], axis[0]])
    origin = anchor + (entry - .90) * axis
    for primitive in (*tile['solids'], *tile['mats']):
        delta = np.array([primitive['x'], primitive['y']]) - origin
        primitive['x'], primitive['y'] = float(delta @ axis), float(delta @ left)
    yaw = math.atan2(-axis[1], axis[0])
    for mat in tile['mats']:
        mat['yaw_rad'] = yaw
    first, last = (tile['solids'][index] for index in (targets[0], targets[-1]))
    far_edge = exit - entry + .90
    tile.update(route_kind=kind, route_targets=targets, route_first_foot=-1,
                route_exit=far_edge, route_entry=.90, route_side=sign,
                start=first['x'] - first['hx'], course_end=last['x'] + last['hx'],
                goal=far_edge + .30)
    return tile


def append_level10(physical_tiles, layout_tiles, seed, settings):
    """Reuse compiled height vectors across independently selected layouts and routes."""
    count = settings.get('height_templates', 8)
    jitter = settings.get('height_jitter_m', .05)
    limit = settings.get('maximum_height_difference_m', .15)
    if count < 2 or count % 2 or not 0 < jitter <= .05 or not 0 < limit <= .15:
        raise ValueError('Invalid height bank size, jitter, or maximum step height')
    rng = np.random.default_rng(seed + 40319)
    physical = list(physical_tiles)
    layouts = list(layout_tiles)
    level9_physical = [tile for tile in physical if tile['level'] == LEVEL - 1]
    level9_layout = [tile for tile in layouts if tile['level'] == LEVEL - 1]
    if len(level9_physical) != 2 or not level9_layout:
        raise ValueError('Level 10 requires both mirrored level 9 yellow courses')
    baseline = np.array([s['height_above_mat_m'] for s in level9_physical[0]['solids']])
    if len(baseline) != 9:
        raise ValueError('Level 10 requires nine physical pillars')

    # A template's exact convex mesh dimensions stay fixed in Warp. Each descriptor
    # binds to one template and moves its meshes/mats to the selected crossing layout.
    for variant in range(count):
        template = len(physical)
        mirror = variant % 2
        heights = _heights(baseline, rng, jitter, limit)
        compiled = _with_heights(level9_physical[mirror], heights)
        compiled.update(level=LEVEL, course_level=LEVEL, variant=variant,
                        origin=[-30. - (template % 8) * 16., -30. - (template // 8) * 8., 0.])
        physical.append(compiled)
        candidates = [tile for tile in level9_layout if tile['variant'] % 2 == mirror]
        # Four independent layouts per height vector include each entry side equally.
        for layout_index in range(4):
            source = candidates[int(rng.integers(len(candidates)))]
            sign = (-1, 1)[layout_index % 2]
            for kind in (LONGITUDINAL, PERPENDICULAR, OBLIQUE):
                for _ in range(16):
                    tile = _with_heights(source, heights)
                    try:
                        tile = _orient(tile, rng, kind, sign)
                    except ValueError:
                        source = candidates[int(rng.integers(len(candidates)))]
                        continue
                    tile.update(level=LEVEL, course_level=LEVEL, variant=len(layouts),
                                physics_template=template, origin=compiled['origin'])
                    layouts.append(tile)
                    break
                else:
                    raise ValueError(f'Failed to build level 10 route {kind}, entry side {sign}')
    return tuple(physical), tuple(layouts)
