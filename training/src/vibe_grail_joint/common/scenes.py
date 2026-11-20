import itertools
import numpy as np

def cases_and_tiles(settings, seed=260027, heights=(.12, .16, .20, .25, .30),
                    speeds=(.25, .4, .7), approaches=(0., 2.), irregular=False, route='up'):
    if route not in ('up', 'down', 'up_down'):
        raise ValueError('Unknown review route')
    rng = np.random.default_rng(seed)
    tiles, cases = [], []
    for height, approach in itertools.product(heights, approaches):
        n = 6
        risers = rng.uniform(.75, 1., n) * height if irregular else np.full(n, height)
        treads = rng.uniform(.28, .40, n) if irregular else np.full(n, .35)
        start = 1.4 + approach
        edges = start + np.r_[0., np.cumsum(treads[:-1])]
        jumps, base = risers.copy(), 0.
        end = start + treads.sum()
        if route == 'down':
            jumps, base = -risers[::-1], float(risers.sum())
        elif route == 'up_down':
            # A complete route with an explicit approach, top landing and exit.
            start = 2.4 + approach
            edges = start + np.r_[0., np.cumsum(treads[:-1])]
            down_start = start + treads.sum() + 1.2
            edges = np.r_[edges, down_start + np.r_[0., np.cumsum(treads[::-1][:-1])]]
            jumps = np.r_[risers, -risers[::-1]]
            end = down_start + treads.sum() + 1.4
        i = len(tiles)
        tiles.append(dict(level=0, variant=0, kind=route, n=n, height=float(risers.max()),
                          nominal_height=height, tread=float(treads.mean()), start=start, plateau=1.2 if route == 'up_down' else 1.,
                          goal=float(end+.6), origin=[-30.-(i%12)*16., -30.-(i//12)*10., 0.],
                          edges=edges.tolist(), jumps=jumps.tolist(), base=base, risers=risers.tolist(),
                          treads=treads.tolist(), total_height=float(risers.sum())))
        for speed, phase in itertools.product(speeds, (0, 5, 10, 15)):
            cases.append(dict(id=len(cases), tile=i, speed=speed,
                              stream=settings['planner_speeds'].index(speed), phase=phase,
                              spawn_xy=[float(rng.uniform(-.2, .2)), float(rng.uniform(-.15, .15))],
                              nominal_height_m=height, approach_extra_m=approach,
                              out_of_training_height=height > .25, route=route,
                              out_of_training_direction=(route != 'up' and settings.get('task_mode') != 'stairs_original_mixture_v1')))
    return tiles, cases
