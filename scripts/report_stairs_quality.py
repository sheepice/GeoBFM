"""Score terminal-truncated, paired G1 stair rollouts under a fixed quality protocol."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def longest(mask):
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    return int((np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)).max(initial=0))


def impact_events(peak, impulse, cfg):
    times = np.flatnonzero((peak > 0) | (impulse > 0))
    groups = np.split(times, np.flatnonzero(np.diff(times) * cfg['control_dt_s'] >= cfg['riser_merge_gap_s']) + 1)
    return sum(1 for idx in groups if idx.size and (
        peak[idx].max() >= cfg['riser_peak_horizontal_n'] or
        impulse[idx].sum(dtype=np.float64) >= cfg['riser_horizontal_impulse_ns']))


KEYS = ('foot_riser_horizontal_peak_n', 'foot_riser_horizontal_impulse_ns',
        'shin_force_n', 'foot_pair_peak_n', 'stable_support', 'sole_edge_margin_m', 'pelvis_yaw_deg', 'active')
FLAGS = ('riser_hit', 'shin_hit', 'foot_pair_hit', 'edge', 'yaw')
LABELS = {'sonic_original':'SONIC', 'grail_mjlab_transfer':'GRAIL（本地 mjlab）',
          'vibe_original':'ViBe 原版', 'vibe_matched':'ViBe-M 4000', 'da2_ours':'DA2 Small 4000'}


def score(a, cfg):
    if not all(np.isfinite(v).all() for v in a.values()):
        raise ValueError('Non-finite quality trace')
    dt = cfg['control_dt_s']
    hits = sum(impact_events(a[KEYS[0]][:, k], a[KEYS[1]][:, k], cfg) for k in range(2))
    support = a['stable_support']
    margin = a['sole_edge_margin_m']
    overhang = support & (margin < -cfg['stable_foot_overhang_m'])
    edge_steps = max(longest(mask) for mask in overhang.T)
    angle = np.abs(a['pelvis_yaw_deg'])
    yaw_steps = longest(angle > cfg['pelvis_abs_yaw_deg'])
    return dict(riser_hit=bool(hits), shin_hit=bool((a['shin_force_n'] > cfg['shin_or_foot_pair_force_n']).any()),
                foot_pair_hit=bool((a['foot_pair_peak_n'] > cfg['shin_or_foot_pair_force_n']).any()),
                edge=edge_steps >= round(cfg['overhang_duration_s']/dt),
                yaw=yaw_steps >= round(cfg['yaw_duration_s']/dt),
                riser_hit_events=hits, longest_overhang_s=edge_steps*dt, longest_yaw_s=yaw_steps*dt,
                worst_stable_overhang_m=max(0., -float(margin[support].min())) if support.any() else None,
                peak_abs_yaw_deg=float(angle.max()))


def verify_inputs(report, reference):
    if len(report['cases']) != 120 or len(reference['cases']) != 120:
        raise ValueError("Expected 120 cases per suite")
    for key in ('route', 'tiles', 'completion_mode', 'success_hold_seconds'):
        if report[key] != reference[key]:
            raise ValueError(f'Mismatched {key}')
    if [r['case'] for r in report['cases']] != [r['case'] for r in reference['cases']]:
        raise ValueError('Mismatched test cases')
    for key in ('root','joints','root_quat','root_velocity','joint_velocity'):
        np.testing.assert_allclose(report['initial_state'][key], reference['initial_state'][key], atol=1e-6, rtol=0)


def read_suite(directory, reference, cfg):
    report = json.loads((directory/'report.json').read_text())
    if report['checkpoint_sha256'] != reference['checkpoint_sha256']:
        raise ValueError(f'Wrong checkpoint for {directory}')
    verify_inputs(report, reference)
    with np.load(directory/'trace.npz') as archive:
        arrays = {key: archive[key] for key in KEYS}
    rows = []
    direction = report['route']
    method = directory.name.removesuffix('_' + direction)
    for idx, (trial, old) in enumerate(zip(report['cases'], reference['cases'], strict=True)):
        end = trial['steps']
        if not arrays['active'][:end, idx].all() or arrays['active'][end:, idx].any():
            raise ValueError('Invalid terminal mask')
        row = dict(method=method, direction=direction, **trial['case'], steps=end,
                   success=trial['success'], historical_success=old['success'],
                   **score({k: v[:end, idx] for k, v in arrays.items()}, cfg))
        row['strict'] = row['success'] and not any(row[k] for k in FLAGS)
        rows.append(row)
    return rows


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    return dict(trials=len(rows), **{k:sum(r[k] for r in rows) for k in ('success','strict',*FLAGS)},
                old_completion_changes=sum(r['success'] != r['historical_success'] for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', type=Path)
    exp = p.parse_args().experiment
    cfg = json.loads((exp/'configs/stairs_quality_v1.json').read_text())
    refs = json.loads((exp/'configs/references.json').read_text())
    rows = []
    groups = []
    by_height = []
    pairs = []
    for direction in ('up','down'):
        reference = None
        suites = {}
        for method in LABELS:
            name = method + '_' + direction
            old = json.loads(Path(refs[name]).read_text())
            actual = json.loads((exp/'evaluation'/name/'report.json').read_text())
            if reference is None:
                reference = actual
            else:
                verify_inputs(actual, reference)
            trials = read_suite(exp/'evaluation'/name, old, cfg)
            rows.extend(trials)
            suites[method] = trials
            groups.append(dict(method=method, direction=direction, **summarize(trials)))
            for height in cfg['heights_cm']:
                subset = [r for r in trials if round(100*r['nominal_height_m']) == height]
                by_height.append(dict(method=method, direction=direction, height_cm=height, **summarize(subset)))
        paired = [(x['strict'],y['strict']) for x,y in zip(suites['da2_ours'],suites['vibe_matched'],strict=True)]
        pairs.append(dict(direction=direction, both_pass=paired.count((True,True)), da2_only=paired.count((True,False)),
                          vibe_m_only=paired.count((False,True)), neither=paired.count((False,False))))
    out = exp/'results'
    for name, table in [('episodes',rows),('counts',groups),('by_height',by_height),('paired_da2_vibe_m',pairs)]:
        write_csv(out/(name+'.csv'), table)
    lines = ['# 固定规则楼梯上的严格通过率', '',
             '每模型每方向 120 回合。表内为回合百分比，分母包含跌倒和超时。各类违规可重叠。', '']
    for d, title in [('up','上楼'),('down','下楼')]:
        lines += ['## '+title,'','| 方法 | 完整通过 ↑ | 严格通过 ↑ | 脚撞立面 ↓ | 小腿/支架撞地形 ↓ | 脚打脚 ↓ | 稳定支撑悬边 ↓ | 持续偏航 ↓ |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in (r for r in groups if r['direction']==d):
            values = [f"{100*row[k]/row['trials']:.1f}" for k in ('success','strict',*FLAGS)]
            lines.append('| '+' | '.join([LABELS[row['method']],*values])+' |')
        lines.append('')
    lines += ['## 配对结果','', '| 方向 | 两者通过 | 仅 DA2 通过 | 仅 ViBe-M 通过 | 两者均未通过 |','|---|---:|---:|---:|---:|']
    for row in pairs:
        lines.append('| '+' | '.join(str(v) for v in row.values())+' |')
    lines += ['', '## 指标与复现','',
              '两种方向均沿世界 +X 轴行走。每个方向评估 5、10、15、20、25 cm，三个速度、两个引道距离、四个 planner 相位。每级踏面 35 cm，共六级，时限 45 s。原始通过要求通过出口并连续 1 s 直立、有脚支撑。所有质量指标的统计窗口从首个动作至原始成功、跌倒或超时，含终止步。严格通过率 = 原始通过且五种违规均未发生的回合数 / 全部回合数。严格违规在离线评分时记失败，仿真继续到原终止以保留可比的通过率及其他违规。','',
              '脚撞立面采用 X2 的水平峰值力 ≥20 N 或事件水平冲量 ≥0.5 N·s。G1 的足部传感器在物理子步识别接触法线 |nz|<0.5 的地形面，取接触槽的最大水平力，每子步累加各槽水平力模长乘时间。同侧脚有力的控制帧间隔 <0.20 s 时并为一次事件。小腿/支架碰地形、脚打脚沿用 G1 原评测的峰值合力 >15 N。','',
              '稳定支撑要求同一脚连续 3 个控制帧检测到踏面支撑。以承重接触位置所在踏面为基准，由鞋底四角求至踏面前、后缘及侧边的最小有符号距离。若同侧脚在稳定支撑时悬出 >2 cm 连续 5 帧（0.10 s），记一次回合违规。此项测量支撑几何；脚真实撞立面由接触传感器另计。骨盆偏航由根部四元数求得，与楼梯前向的绝对夹角 >20° 连续 10 帧（0.20 s）即违规。两项为本次 G1 评测新增。','',
              '本任务使用原有前进速度命令，不考核 X2 的终点定点停步。GRAIL 为历史本地 mjlab 策略迁移条件。结果为指定检查点和固定测试集，训练种子差异未评估。配置在开跑前确定。','',
              '各模型与历史场景及权重 SHA256 核对；同方向五模型核对逐项 case、地形和初始物理状态（绝对容差 1e-6）。',
              f"与历史记录相比，原始通过结果逐回合有 {sum(r['old_completion_changes'] for r in groups)}/1200 处不同。表中完整通过率取本次重跑。",
              '', '逐回合见 [episodes.csv](episodes.csv)，分高度计数见 [by_height.csv](by_height.csv)，原始计数见 [counts.csv](counts.csv)。']
    (out/'stairs_quality.md').write_text('\n'.join(lines)+'\n')
    print(out/'stairs_quality.md')


if __name__ == '__main__':
    main()
