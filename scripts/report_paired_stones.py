"""Validate paired cylinder evaluation outputs and summarize finite-set metrics."""
import argparse
from collections import Counter
import csv
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean, median


MODELS = ('vibe_matched', 'da2_ours')
OUTCOMES = ('success', 'floor', 'fall_or_out_of_bounds', 'stall', 'timeout')


def write_csv(path, rows):
    with path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    counts = Counter(row['outcome'] for row in rows)
    events = {key: counts[key] for key in OUTCOMES}
    events.update(clean_success=sum(r['clean_success'] for r in rows),
                  wall=sum(r['wall_peak_n'] > 15 for r in rows),
                  pair=sum(r['pair_peak_n'] > 15 for r in rows),
                  shin=sum(r['shin_peak_n'] > 15 for r in rows))
    return dict(n=len(rows), **events, **{k+'_rate': v/len(rows) for k, v in events.items()},
                mean_progress=mean(r['completed_pillars']/r['total_pillars'] for r in rows))


def report(experiment):
    evaluation, out = experiment/'evaluation', experiment/'results'
    seeds = sorted(int(p.name.split('_')[-1]) for p in evaluation.glob('vibe_matched_*'))
    if not seeds:
        raise ValueError('Missing paired seeds')
    expected = list(product(range(7), range(4), (0, 5, 10, 15), (-.1, .1)))
    episodes = []
    pairs = Counter()
    matched_times = {model: [] for model in MODELS}
    initial_errors = {}
    checkpoints = {}
    for seed in seeds:
        initial, outcomes = [], []
        for model in MODELS:
            directory = evaluation/f'{model}_{seed}'
            initial.append(json.loads((directory/'initial.json').read_text()))
            data = json.loads((directory/'report.json').read_text())
            assert data['iteration'] == 4000 and data['method'] == model and data['seed'] == seed
            assert data['control_dt'] == .02 and data['physics_dt'] == .005
            assert data['sha256'] == checkpoints.setdefault(model, data['sha256'])
            rows = data['episodes']
            assert len(rows) == len(expected)
            for i, row in enumerate(rows):
                assert row['case'] == i and (row['level'], row['variant'], row['phase'], row['offset_m']) == expected[i]
                assert row['outcome'] in OUTCOMES and 0 < row['seconds'] <= 35.
                assert row['total_pillars'] == 6 + 2 * row['level']
                assert 0 <= row['completed_pillars'] <= row['total_pillars']
                success = row['outcome'] == 'success'
                assert not success or row['completed_pillars'] == row['total_pillars']
                peaks = [row[k+'_peak_n'] for k in ('wall', 'pair', 'shin')]
                assert all(math.isfinite(p) and p >= 0 for p in peaks)
                assert row['clean_success'] == (success and max(peaks) <= 15)
                episodes.append(dict(model=model, seed=seed, **row))
            outcomes.append(rows)
        a, b = initial
        assert a['tiles'] == b['tiles'] and len(a['states']) == len(expected)
        error = max(abs(x-y) for s,t in zip(a['states'], b['states'], strict=True)
                    for x,y in zip(s,t,strict=True))
        assert error < 1e-6
        initial_errors[seed] = error
        for left, right in zip(*outcomes, strict=True):
            vibe, da2 = (r['outcome'] == 'success' for r in (left, right))
            key = ('both' if da2 else 'vibe_only') if vibe else ('da2_only' if da2 else 'neither')
            pairs[key] += 1
            if vibe and da2:
                for model, row in zip(MODELS, (left, right), strict=True):
                    matched_times[model].append(row['seconds'])
    write_csv(out/'episodes.csv', episodes)
    tables = {}
    for name, groups in [('overall', [None]), ('level', list(range(7))), ('seed', seeds)]:
        table = []
        for value in groups:
            for model in MODELS:
                rows = [r for r in episodes if r['model'] == model and (name == 'overall' or r[name] == value)]
                group = {} if name == 'overall' else {name: value}
                table.append(dict(model=model, **group, **summarize(rows)))
        write_csv(out/f'{name}.csv', table)
        tables[name] = table
    result = dict(status='completed', seeds=seeds, checkpoint_sha256=checkpoints,
                  max_initial_state_difference=initial_errors,
                  paired_success={k: pairs[k] for k in ('both','vibe_only','da2_only','neither')},
                  median_seconds_on_shared_success={m: median(v) if v else None for m,v in matched_times.items()},
                  **tables)
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    args = parser.parse_args()
    result = report(args.experiment)
    print(json.dumps(result['overall'], indent=2))
