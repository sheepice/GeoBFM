"""Validate the hard pillar factorial and report paired checkpoint differences."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from report_paired_stones import MODELS, OUTCOMES, summarize, write_csv


def geometry(tiles, protocol):
    rows = []
    for i, nominal in enumerate(tiles['nominal']):
        level = nominal['level']
        actual = {}
        for condition, courses in tiles.items():
            tile = courses[i]
            assert all(tile[key] == nominal[key] for key in
                       ('terrain_seed', 'level', 'variant', 'origin', 'n', 'stagger_m', 'start'))
            stones = tile['solids']
            baseline = nominal['solids']
            assert len(stones) == 6 + 2*level
            assert stones[0] == baseline[0]
            for a, b in zip(stones, baseline, strict=True):
                assert all(a[key] == b[key] for key in ('hx','hy','y','shape','collision_sides'))
            xyz = np.array([[s['x'], s['y'], s['top']] for s in stones])
            radii = np.array([s['hx'] for s in stones])
            gap = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1) - radii[:-1] - radii[1:]
            rise = np.diff(xyz[:, 2])
            assert np.allclose(gap, tile['surface_gaps_m'], atol=1e-12, rtol=0)
            assert np.isclose(xyz[-1,0] + radii[-1] + 1., tile['goal'])
            if condition in ('height', 'joint'):
                bound = protocol['max_adjacent_abs_height_difference_m'][str(level)]
                minimum, maximum = protocol['top_height_bounds_m']
                assert max(abs(rise)) <= bound and np.all((xyz[:,2] >= minimum) & (xyz[:,2] <= maximum))
            if condition in ('gap', 'joint'):
                minimum, maximum = protocol['horizontal_edge_gap_range_m'][str(level)]
                assert np.all((gap >= minimum) & (gap <= maximum))
            actual[condition] = xyz
            rows.append(dict(condition=condition, terrain_seed=tile['terrain_seed'], level=level,
                variant=tile['variant'], max_abs_rise_m=max(abs(rise)), mean_abs_rise_m=np.mean(abs(rise)),
                min_rise_m=min(rise), max_rise_m=max(rise), min_gap_m=min(gap), max_gap_m=max(gap),
                mean_gap_m=np.mean(gap)))
        for a, b, axis in [('nominal','height',0), ('nominal','gap',2),
                           ('gap','joint',0), ('height','joint',2)]:
            assert np.array_equal(actual[a][:,axis], actual[b][:,axis])
        assert not np.array_equal(actual['nominal'][:,2], actual['joint'][:,2])
        assert not np.array_equal(actual['nominal'][:,0], actual['joint'][:,0])
    return rows


def report(exp):
    p = json.loads((exp/'configs/protocol.json').read_text())
    expected = list(product(p['terrain_seeds'], p['levels'], range(p['variants_per_seed_and_level']),
                            p['phases'], p['start_offsets_m']))
    keys = ('terrain_seed', 'level', 'variant', 'phase', 'offset_m')
    out = exp/'results'
    episodes, summary, by_level, pair_counts = [], [], [], []
    nominal_states = None
    max_state_error = 0.
    tiles, differences, fingerprints = {}, {}, {}
    for condition in p['conditions']:
        model_rows = []
        for model in MODELS:
            directory = exp/'evaluation'/f'{condition}_{model}'
            initial = json.loads((directory/'initial.json').read_text())
            assert tiles.setdefault(condition, initial['tiles']) == initial['tiles']
            states = np.asarray(initial['states'])
            if nominal_states is None:
                nominal_states = states
            assert states.shape == nominal_states.shape and len(states) == len(expected)
            max_state_error = max(float(np.max(abs(states - nominal_states))), max_state_error)
            assert max_state_error < 1e-6
            data = json.loads((directory/'report.json').read_text())
            assert data['sha256'] == fingerprints.setdefault(model, data['sha256'])
            assert data['geometry_condition'] == condition and data['method'] == model
            assert data['seed'] == p['simulation_seed'] and data['iteration'] == 4000
            assert data['control_dt'] == .02 and data['physics_dt'] == .005
            rows = data['episodes']
            for i,(row, coordinates) in enumerate(zip(rows, expected, strict=True)):
                assert row['case'] == i and tuple(row[key] for key in keys) == coordinates
                assert row['outcome'] in OUTCOMES and 0 < row['seconds'] <= p['timeout_s']
                assert 0 <= row['completed_pillars'] <= row['total_pillars'] == 6 + 2*row['level']
                ok = row['outcome'] == 'success'
                assert not ok or row['completed_pillars'] == row['total_pillars']
                peaks = np.array([row[key+'_peak_n'] for key in ('wall','pair','shin')])
                assert np.all(np.isfinite(peaks) & (peaks >= 0))
                assert row['clean_success'] == (ok and max(peaks) <= p['force_threshold_n'])
                episodes.append(dict(condition=condition, model=model, **row))
            model_rows.append(rows)
            summary.append(dict(condition=condition, model=model, **summarize(rows)))
            for level in p['levels']:
                by_level.append(dict(condition=condition, model=model, level=level,
                                     **summarize([row for row in rows if row['level']==level])))
        successes = np.array([[r['outcome'] == 'success' for r in rows] for rows in model_rows])
        vibe, da2 = successes
        differences[condition] = da2.astype(int)-vibe.astype(int)
        pair_counts.append(dict(condition=condition, both=int(sum(vibe & da2)),
            vibe_only=int(sum(vibe & ~da2)), da2_only=int(sum(da2 & ~vibe)),
            neither=int(sum(~vibe & ~da2)), da2_minus_vibe_pp=100*np.mean(differences[condition])))
    write_csv(out/'layouts.csv', geometry(tiles,p))
    for filename, table in [('overall',summary), ('level',by_level), ('paired',pair_counts), ('episodes',episodes)]:
        write_csv(out/(filename+'.csv'), table)

    # Keep all eight starts of each route together. Resample terrain seeds within each level/variant.
    def interval(values):
        shape = (len(p['terrain_seeds']), len(p['levels'])*p['variants_per_seed_and_level'], -1)
        routes = values.reshape(shape).mean(axis=2).T
        rng = np.random.default_rng(481101)
        index = rng.integers(routes.shape[1], size=(10000,*routes.shape))
        draws = routes[np.arange(len(routes))[None,:,None],index].mean(axis=(1,2))*100
        return dict(estimate_pp=float(values.mean()*100), layout_bootstrap_95_pp=np.quantile(draws,[.025,.975]).tolist())

    result = dict(status='completed', fingerprint=fingerprints, max_initial_state_difference=max_state_error,
        joint_success_advantage=interval(differences['joint']),
        joint_minus_nominal_success_advantage=interval(differences['joint']-differences['nominal']),
        paired=pair_counts, overall=summary, by_level=by_level)
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    result = report(parser.parse_args().experiment)
    print(json.dumps(result['paired'], indent=2))
