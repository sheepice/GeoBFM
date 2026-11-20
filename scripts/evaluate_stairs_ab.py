#!/usr/bin/env python3
"""Evaluate both final checkpoints on predeclared, paired held-out stair cases."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


METHODS = ('vibe_matched', 'da2_ours')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def initial_audit(a, b, rgb_a=None, rgb_b=None):
    import numpy as np
    if set(a) != set(b): raise ValueError('Unmatched initial state fields')
    maximum = 0.
    for key in a:
        if key == 'rgb_sha256': continue
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape or not np.allclose(x, y, rtol=0, atol=1e-6):
            raise ValueError(f'Unmatched initial physics: {key}')
        maximum = max(maximum, float(np.max(np.abs(x-y), initial=0.)))
    rgb_max, rgb_mean = 0, 0.
    if a['rgb_sha256'] != b['rgb_sha256'] or (rgb_a is not None and rgb_b is not None):
        if rgb_a is None or rgb_b is None or rgb_a.shape != rgb_b.shape:
            raise ValueError('Unmatched initial RGB; pixel arrays required for numerical audit')
        difference = np.abs(rgb_a.astype(np.int16)-rgb_b.astype(np.int16))
        rgb_max, rgb_mean = int(difference.max(initial=0)), float(difference.mean())
        if rgb_max > 1 or rgb_mean > .01:
            raise ValueError(f'Unmatched initial RGB beyond quantization tolerance: max={rgb_max}, mean={rgb_mean}')
    return dict(maximum_physics_absolute_difference=maximum, maximum_rgb_uint8_difference=rgb_max,
                mean_rgb_uint8_difference=rgb_mean)


def paired_reports(a, b, rgb_a=None, rgb_b=None):
    """Refuse scores from different terrain or initial conditions (float tolerance 1e-6)."""
    for key in ('tiles', 'completion_mode', 'success_hold_seconds'):
        if a[key] != b[key]:
            raise ValueError(f'Unmatched evaluation: {key}')
    initial_audit(a['initial_state'], b['initial_state'], rgb_a, rgb_b)
    if len(a['cases']) != len(b['cases']):
        raise ValueError('Different case counts')
    wins = dict(both_success=0, da2_only=0, vibe_only=0, both_failure=0)
    for ra, rb in zip(a['cases'], b['cases'], strict=True):
        if ra['case'] != rb['case']:
            raise ValueError('Different case specifications')
        key = ('both_success' if ra['success'] and rb['success'] else
               'da2_only' if rb['success'] else 'vibe_only' if ra['success'] else 'both_failure')
        wins[key] += 1
    return wins


def summarize(output):
    import csv
    import numpy as np
    protocol = json.loads((output / 'protocol.json').read_text())
    rows, pairs = [], []
    for suite in protocol['suites']:
        reports = {method: json.loads((output / f'{method}_{suite["name"]}' / 'report.json').read_text())
                   for method in METHODS}
        image_paths = [output/f'{method}_{suite["name"]}'/'initial_rgb.npy' for method in METHODS]
        images = [np.load(p) if p.exists() else None for p in image_paths]
        wins = paired_reports(reports[METHODS[0]], reports[METHODS[1]], *images)
        pairs.append(dict(suite=suite['name'], **wins,
            initial_audit=initial_audit(reports[METHODS[0]]['initial_state'], reports[METHODS[1]]['initial_state'], *images)))
        for method, report in reports.items():
            for row in report['cases']:
                rows.append(dict(method=method, suite=suite['name'], route=suite['route'],
                                 irregular=suite['irregular'], **row))
    metrics = []
    for method in METHODS:
        for height in [None, *protocol['heights_m']]:
            for route in [None, *protocol.get('routes', ['up', 'down', 'up_down'])]:
                for domain in ('all', 'in_training_height', 'above_training_height'):
                    selected = [r for r in rows if r['method'] == method
                                and (height is None or r['case']['nominal_height_m'] == height)
                                and (route is None or r['route'] == route)
                                and (domain == 'all' or r['case']['out_of_training_height'] ==
                                     (domain == 'above_training_height'))]
                    if not selected:
                        continue
                    successful = [r for r in selected if r['success']]
                    mean = lambda key, data=selected: sum(r[key] for r in data) / len(data) if data else None
                    metrics.append(dict(method=method, height_m=height, route=route, domain=domain,
                        trials=len(selected), successes=len(successful), success_rate=len(successful)/len(selected),
                        fall_rate=mean('fall'), collision_free_success_rate=sum(r['success'] and r['collision_free']
                                                                                 for r in selected)/len(selected),
                        mean_foot_strikes=mean('foot_strikes'), mean_shin_strikes=mean('shin_strikes'),
                        foot_riser_collision_rate=sum(r['foot_strikes'] > 0 for r in selected)/len(selected),
                        shin_collision_rate=sum(r['shin_strikes'] > 0 for r in selected)/len(selected),
                        mean_foot_riser_impulse_ns=mean('foot_impulse_ns') if 'foot_impulse_ns' in selected[0] else None,
                        mean_shin_impulse_ns=mean('shin_impulse_ns') if 'shin_impulse_ns' in selected[0] else None,
                        mean_speed_error_m_s=mean('speed_error_m_s'),
                        mean_success_time_s=mean('time_s', successful),
                        mean_completion_fraction=mean('completion_fraction') if 'completion_fraction' in selected[0] else None,
                        mean_failure_completion_fraction=mean('completion_fraction', [r for r in selected if not r['success']])
                            if 'completion_fraction' in selected[0] else None,
                        foot_pair_collision_rate=mean('foot_pair_collision') if 'foot_pair_collision' in selected[0] else None,
                        mean_foot_pair_events=mean('foot_pair_events') if 'foot_pair_events' in selected[0] else None))
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    write_json(output / 'summary.json', dict(metrics=metrics, paired_outcomes=pairs,
        initial_state_and_rgb_matched_with_numeric_tolerances=True, training_seeds=1,
        note='One training seed per method. Case-level differences do not establish multi-seed superiority.'))
    write_json(output / 'episodes.json', rows)
    flat_rows = [{**{k: v for k, v in row.items() if k != 'case'}, **row['case']} for row in rows]
    with (output / 'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0])); writer.writeheader(); writer.writerows(flat_rows)
    names = dict(vibe_matched='ViBe-M', da2_ours='DA2')
    find = lambda method, height=None, route=None, domain='all': next(
        r for r in metrics if (r['method'], r['height_m'], r['route'], r['domain']) == (method, height, route, domain))
    checkpoint_index = protocol.get('checkpoint_index', 4000)
    lines = [f'# 检查点 {checkpoint_index}：ViBe-M / DA2 固定场景评测', '',
        f'两组均使用预先指定的 `model_{checkpoint_index}.pt`，不依据测试成绩选择权重。每组一个训练种子，'
        f'{len(rows)//2} 个回合；方向、速度和几何参数详见 `protocol.json`。包含规则/不规则楼梯、两个引道距离和四个 planner 相位。', '',
        '配对核查已通过：每组场景的楼梯参数相同；初始根部/关节位姿和速度绝对误差不超过 1e-6，'
        '初始 RGB 最多差 1 个 uint8 灰度级、平均误差不超过 0.01 灰度级。逐场景实测误差保存在 `summary.json`。'
        '使用确定性策略动作，不加载训练优化器、教师监督或几何标签；DA2 几何辅助头移除前后动作完全相同。', '',
        '## 主结果', '',
        '| 范围 | 方法 | 成功 / 总数 | 通过率 | 跌倒率 | 无台阶碰撞通过率 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for domain, label in [('in_training_height', '5–25 cm'), ('above_training_height', '30 cm（超训练范围）'), ('all', '全部')]:
        if not any(r['domain'] == domain for r in metrics): continue
        for method in METHODS:
            r = find(method, domain=domain)
            lines.append(f'| {label} | {names[method]} | {r["successes"]}/{r["trials"]} | '
                         f'{r["success_rate"]:.2%} | {r["fall_rate"]:.2%} | {r["collision_free_success_rate"]:.2%} |')
    lines += ['', '## 按高度统计', '',
              '| 台阶高度 | ViBe-M | DA2 | DA2 − ViBe-M（百分点） |', '| --- | ---: | ---: | ---: |']
    for height in protocol['heights_m']:
        a, b = [find(method, height=height) for method in METHODS]
        lines.append(f'| {height*100:.0f} cm | {a["successes"]}/{a["trials"]} ({a["success_rate"]:.2%}) | '
                     f'{b["successes"]}/{b["trials"]} ({b["success_rate"]:.2%}) | '
                     f'{100*(b["success_rate"]-a["success_rate"]):+.2f} |')
    lines += ['', '## 按方向统计（5–25 cm）', '', '| 方向 | ViBe-M | DA2 |', '| --- | ---: | ---: |']
    for route, label in [('up', '上楼'), ('down', '下楼'), ('up_down', '先上后下')]:
        if route not in protocol.get('routes', ['up', 'down', 'up_down']): continue
        a, b = [find(method, route=route, domain='in_training_height') for method in METHODS]
        lines.append(f'| {label} | {a["success_rate"]:.2%} ({a["successes"]}/{a["trials"]}) | '
                     f'{b["success_rate"]:.2%} ({b["successes"]}/{b["trials"]}) |')
    lines += ['', '## 配对结果', '',
              '| 场景 | 两者成功 | 仅 DA2 成功 | 仅 ViBe-M 成功 | 两者失败 |', '| --- | ---: | ---: | ---: | ---: |']
    for pair in pairs:
        lines.append('| ' + ' | '.join(str(pair[k]) for k in ('suite', 'both_success', 'da2_only', 'vibe_only', 'both_failure')) + ' |')
    if 'foot_pair_collision' in rows[0]:
        lines += ['', '## 动作质量（全部回合）', '',
            '| 方法 | 平均完成比例 | 失败回合平均完成比例 | 脚打脚回合率 | 脚踢立面回合率 | 小腿碰撞回合率 | 成功回合平均耗时 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for method in METHODS:
            r = find(method)
            failure_progress = f'{r["mean_failure_completion_fraction"]:.2%}' if r['mean_failure_completion_fraction'] is not None else '无失败回合'
            success_time = f'{r["mean_success_time_s"]:.2f} s' if r['mean_success_time_s'] is not None else '无成功回合'
            lines.append(f'| {names[method]} | {r["mean_completion_fraction"]:.2%} | {failure_progress} | '
                         f'{r["foot_pair_collision_rate"]:.2%} | {r["foot_riser_collision_rate"]:.2%} | '
                         f'{r["shin_collision_rate"]:.2%} | {success_time} |')
    lines += ['', '## 判定与局限', '',
        '通过要求机器人根部越过场景出口、双脚越过出口前 45 cm，连续 1 秒保持直立、至少一脚支撑；'
        '上/下楼超时 45 秒，先上后下超时 60 秒。跌倒为重力投影 z > −0.4 或根部离地 < 25 cm。'
        '碰撞为接触观测器记录的脚踢立面或小腿撞地形超过 15 N；无碰撞通过比普通通过更严格。', '',
        '不规则场景每级高度为标称值的 '
        '75%–100%，踏面 28–40 cm；规则场景每级等于标称高度、踏面 35 cm。', '',
        '这是一个训练种子的初步方法对比，不能证明多种子意义上的显著优越性，也不能据此推断真机鲁棒性。'
        '对手是共用楼梯 PPO 配方的 ViBe-M，并非原版 ViBe 全训练流程。没有测试光照/颜色扰动。', '',
        '完整协议：`protocol.json`；每回合数据：`episodes.csv` / `episodes.json`；'
        '分高度、方向、范围汇总：`summary.csv`；轨迹与权重 SHA256：各场景 `report.json` / `trace.npz`。', '']
    (output / 'report.md').write_text('\n'.join(lines))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--seed', type=int, default=431113)
    parser.add_argument('--checkpoint-index', type=int, default=4000)
    parser.add_argument('--heights', default='.05,.10,.15,.20,.25')
    parser.add_argument('--speeds', default='.25,.4,.7')
    parser.add_argument('--routes', default='up,down,up_down')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.summarize_only:
        summarize(output)
        return
    experiment = args.experiment.resolve()
    output.mkdir(parents=True, exist_ok=False)
    gpus = [int(x) for x in args.gpus.split(',')]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError('Select distinct nonnegative GPUs')
    routes = args.routes.split(',')
    if not routes or set(routes) - {'up', 'down', 'up_down'}: raise ValueError('Invalid routes')
    heights = [float(x) for x in args.heights.split(',')]
    speeds = [float(x) for x in args.speeds.split(',')]
    suites = [dict(name=f'{route}_{"irregular" if irregular else "regular"}', route=route,
                   irregular=irregular, duration_s=60 if route == 'up_down' else 45)
              for route in routes for irregular in (False, True)]
    protocol = dict(version=1, checkpoint_index=args.checkpoint_index, seed=args.seed,
        methods=list(METHODS), heights_m=heights, speeds_m_s=speeds, routes=routes,
        extra_approach_m=[0., 2.], planner_phases=[0, 5, 10, 15], suites=suites,
        success='Reach exit with both feet past goal-0.45m, upright supported walking for 1 continuous second',
        fall='Projected gravity z > -0.4 or root-to-ground clearance < 0.25m',
        domain='Nominal physics/camera/light; no observation noise or domain perturbations; deterministic actions',
        matching='Identical tiles/cases; initial physics abs error <=1e-6; initial RGB max <=1 uint8 level and mean <=0.01',
        episodes_per_method=len(suites)*len(heights)*len(speeds)*2*4,
        foot_pair_events='Physics-substep actual force > 15N; contiguous control steps form one event; measurement only',
        completion_fraction='Maximum fraction of risers passed by both foot centers and pelvis while upright and supported, before terminal',
        checkpoint_selection=f'Fixed model_{args.checkpoint_index}; no selection using test outcomes',
        repo=str(Path(__file__).resolve().parents[1]), experiment=str(experiment))
    protocol['source_sha256'] = {str(p.relative_to(Path(__file__).resolve().parents[1])):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__), *[Path(__file__).resolve().parents[1] / 'training/src/vibe_grail_joint/da2_locomotion' / name
                           for name in ('evaluate.py', 'scenes.py')]]}
    write_json(output / 'protocol.json', protocol)
    jobs = deque((method, suite) for suite in suites for method in METHODS)
    free = deque(gpus); active = []; records = []
    while jobs or active:
        while jobs and free:
            method, suite = jobs.popleft(); gpu = free.popleft()
            name = f'{method}_{suite["name"]}'
            command = [sys.executable, 'scripts/run.py', 'stairs', 'evaluate', '--config',
                str(experiment/'configs'/f'stairs_ab_{method}_sonic_raw_20261112.json'), '--gpus', str(gpu),
                '--checkpoint', str(experiment/'runs'/f'{method}_seed291029'/f'model_{args.checkpoint_index}.pt'),
                '--output', str(output/name), '--seed', str(args.seed), '--heights', args.heights,
                '--speeds', args.speeds, '--approaches', '0,2', '--route', suite['route'],
                '--duration', str(suite['duration_s']), '--completion-mode', 'walking', '--hold-seconds', '1',
                '--match-height-initialization', '--measure-foot-contact']
            if suite['irregular']: command.append('--irregular')
            log = (output/f'{name}.log').open('w')
            process = subprocess.Popen(command, cwd=protocol['repo'], stdout=log, stderr=subprocess.STDOUT)
            record = dict(name=name, method=method, suite=suite['name'], gpu=gpu, pid=process.pid,
                          command=command, status='running')
            active.append((process, record, log)); records.append(record)
            print(json.dumps(dict(started=name, gpu=gpu, pid=process.pid)), flush=True)
        for entry in list(active):
            process, record, log = entry
            if process.poll() is None: continue
            log.close(); record.update(exit_code=process.returncode,
                                      status='completed' if process.returncode == 0 else 'failed')
            active.remove(entry); free.append(record['gpu'])
            print(json.dumps(dict(finished=record['name'], exit_code=process.returncode)), flush=True)
        write_json(output/'launches.json', records)
        if jobs or active: time.sleep(5)
    if any(r['status'] != 'completed' for r in records):
        raise RuntimeError('Some evaluation jobs failed; inspect launches.json and job logs')
    summarize(output)
    print(json.dumps(dict(status='completed', output=str(output))), flush=True)


if __name__ == '__main__':
    main()
