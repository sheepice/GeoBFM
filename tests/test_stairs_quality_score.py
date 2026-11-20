import json
from pathlib import Path

import numpy as np

from report_stairs_quality import KEYS, impact_events, score


CFG = json.loads((Path(__file__).resolve().parents[1] / 'configs/evaluation/stairs_quality_v1.json').read_text())


def test_same_foot_and_consecutive_yaw_are_required():
    a = {key: np.zeros((20, 2)) for key in KEYS}
    a['stable_support'] = np.ones((20, 2), dtype=bool)
    a['pelvis_yaw_deg'] = np.zeros(20)
    a['sole_edge_margin_m'][:4, 0] = -0.021
    a['sole_edge_margin_m'][4:8, 1] = -0.021
    a['pelvis_yaw_deg'][:9] = -21
    a['pelvis_yaw_deg'][10:19] = 21
    assert not score(a, CFG)['edge'] and not score(a, CFG)['yaw']
    a['sole_edge_margin_m'][4, 0] = -0.021
    a['pelvis_yaw_deg'][9] = -21
    assert score(a, CFG)['edge'] and score(a, CFG)['yaw']
    a['stable_support'][:] = False
    assert not score(a, CFG)['edge']


def test_peak_or_impulse_and_episode_separation():
    peak, impulse = np.zeros(24), np.zeros(24)
    peak[[1, 10, 20]] = [5., 5., 20.]
    impulse[[1, 10, 20]] = [.25, .25, .01]
    assert impact_events(peak, impulse, CFG) == 2
    impulse[10] = .24
    assert impact_events(peak, impulse, CFG) == 1
