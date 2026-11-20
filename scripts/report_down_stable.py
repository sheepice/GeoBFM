"""Rescore descent from the paired regular-stairs quality evaluation."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, help='Directory containing the quality v1 episodes.csv')
    results = parser.parse_args().results
    groups = {}
    with (results / 'episodes.csv').open(newline='') as source:
        for row in csv.DictReader(source):
            if row['direction'] != 'down':
                continue
            flags = {k: {'True': True, 'False': False}[row[k]]
                     for k in ('success', 'strict', 'riser_hit', 'shin_hit', 'foot_pair_hit', 'edge', 'yaw')}
            stable = flags['success'] and not any(flags[k] for k in ('shin_hit', 'foot_pair_hit', 'edge', 'yaw'))
            if flags['strict'] != (stable and not flags['riser_hit']):
                raise ValueError(f"Original strict score mismatch for {row['method']}, case {row['id']}")
            cases = groups.setdefault(row['method'], {})
            case_id = int(row['id'])
            if case_id in cases:
                raise ValueError(f'Duplicate case {case_id}')
            cases[case_id] = (flags['success'], flags['strict'], stable, flags['riser_hit'], stable and flags['riser_hit'])

    labels = {'sonic_original': 'SONIC 原版', 'grail_mjlab_transfer': 'GRAIL（本地 mjlab）',
              'vibe_original': 'ViBe 原版', 'vibe_matched': 'ViBe-M 4000', 'da2_ours': 'DA2 Small 4000'}
    if groups.keys() != labels.keys() or any(set(v) != set(range(120)) for v in groups.values()):
        raise ValueError('Expected all five models and the complete 120-case descent grid')

    lines = ['# 规则下楼稳定通过率（补充分析）', '',
             '复用原评测的 600 个下楼回合。该口径在查看首次严格评测后提出，所有模型共用同一判据。', '',
             '稳定通过要求在 45 s 内完成原下楼任务，且整个回合没有脚打脚、小腿或支架撞地形（峰值合力 >15 N）、同侧稳定承重脚持续悬出踏面（>2 cm、连续 ≥0.10 s），以及骨盆持续偏航（>20°、连续 ≥0.20 s）。',
             '脚撞立面作为独立指标。水平峰值力 ≥20 N 或单次水平冲量 ≥0.5 N·s 即计撞击；新口径允许此类回合在满足其余条件时通过，包括较重的撞击。原严格通过仍要求全程没有这类撞击。每列以全部 120 回合为分母。', '',
             '| 模型 | 完整通过 | 原严格通过 | 下楼稳定通过 | 脚撞立面 | 稳定通过且撞立面 |',
             '|---|---:|---:|---:|---:|---:|']
    with (results / 'down_stable_counts.csv').open('w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['method', 'trials', 'completed', 'strict_v1', 'stable_down', 'riser_hit', 'stable_with_riser_hit'])
        for method, label in labels.items():
            counts = [sum(column) for column in zip(*groups[method].values())]
            writer.writerow([method, 120, *counts])
            cells = [f'{n}/120 ({100*n/120:.1f}%)' for n in counts]
            lines.append('| ' + ' | '.join([label, *cells]) + ' |')
    (results / 'down_stable.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
