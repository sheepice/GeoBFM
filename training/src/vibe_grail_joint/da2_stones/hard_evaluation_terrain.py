"""Paired high-level pillar geometry for evaluation only."""
from copy import deepcopy
import math

import numpy as np

from .terrain import make_tiles


CONDITIONS = ('nominal', 'height', 'gap', 'joint')
SEEDS = (481119, 481127, 481133, 481139)
HEIGHT_STEP = {4: .08, 5: .10, 6: .12}
GAP_MAX = {4: .17, 5: .20, 6: .23}


def hard_tiles(condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    tiles = []
    for seed in SEEDS:
        for base in make_tiles(seed):
            level, variant = base['level'], base['variant']
            if level not in HEIGHT_STEP:
                continue
            tile = deepcopy(base)
            stones = tile['solids']
            height_rng, gap_rng = [np.random.default_rng([seed, level, variant, axis]) for axis in (1, 2)]
            for i, (left, right) in enumerate(zip(stones, stones[1:])):
                if condition in ('height', 'joint'):
                    limit = HEIGHT_STEP[level]
                    right['top'] = height_rng.uniform(max(.06, left['top'] - limit),
                                                      min(.34, left['top'] + limit))
                if condition in ('gap', 'joint'):
                    gap = gap_rng.uniform(.04, GAP_MAX[level])
                    dy = right['y'] - left['y']
                    distance = left['hx'] + right['hx'] + gap
                    right['x'] = left['x'] + math.sqrt(distance**2 - dy**2)
                    tile['surface_gaps_m'][i] = float(gap)
            end = stones[-1]['x'] + stones[-1]['hx']
            index = len(tiles)
            tile.update(terrain_seed=seed, course_end=end, goal=end + 1.,
                        height=max(s['top'] for s in stones),
                        tread=(stones[-1]['x'] - stones[0]['x']) / (len(stones) - 1),
                        origin=[-30. - (index % 8) * 16., -30. - (index // 8) * 8., 0.])
            tiles.append(tile)
    return tuple(tiles)
