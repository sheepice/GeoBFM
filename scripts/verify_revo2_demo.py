#!/usr/bin/env python3
"""Audit a saved Revo2 demo without stepping physics or running a policy."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def verify(run):
    run = Path(run)
    trace = json.loads((run / 'trace.json').read_text())
    config = json.loads((run / 'configuration.json').read_text())
    poses = np.load(run / 'trajectory.npz')['qpos']
    model = mujoco.MjModel.from_binary_path(str(run / 'scene.mjb'))
    data = mujoco.MjData(model)
    if len(trace) != len(poses):
        raise ValueError('Trace and trajectory lengths differ')
    times = np.array([row['t'] for row in trace])
    roots = np.array([row['root'] for row in trace])
    objects = np.array([row['object_position'] for row in trace])
    phases = np.array([row['phase'] for row in trace])
    commands = np.array([row['speed_command'] for row in trace])
    contacts = [row['object_contacts'] for row in trace]
    carry = np.isin(phases, ['CARRY_UPSTAIRS', 'COMPLETE'])
    final = times >= times[-1] - 3
    landing = model.geom('landing')
    bounds = np.stack([landing.pos - landing.size, landing.pos + landing.size])
    terrain = [model.geom(f'step_{i}') for i in range(6)] + [landing]
    ground = np.zeros(len(roots))
    for geom in terrain:
        inside = np.all(np.abs(roots[:, :2] - geom.pos[:2]) <= geom.size[:2], axis=1)
        ground[inside] = np.maximum(ground[inside], geom.pos[2] + geom.size[2])
    foot_ids = [model.body(f'{side}_ankle_roll_link').id for side in ('left', 'right')]
    foot_positions, foot_contacts = [], []
    for pose in poses[final]:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        foot_positions.append(data.xpos[foot_ids].copy())
        touching = set()
        for contact in data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if a == landing.id or b == landing.id:
                other = b if a == landing.id else a
                touching.add(int(model.geom_bodyid[other]))
        foot_contacts.append([bid in touching for bid in foot_ids])
    foot_positions = np.asarray(foot_positions)
    foot_contacts = np.asarray(foot_contacts)
    transitions = []
    for i, row in enumerate(trace):
        if i == 0 or phases[i] != phases[i - 1]:
            transitions.append(dict(t_s=row['t'], phase=row['phase']))
    hand_contact = np.array([any(n.startswith('revo_right_') for n in c) for c in contacts])
    nonhand_support = np.array([any(not n.startswith('revo_right_') for n in c) for c in contacts])
    separation = np.linalg.norm(objects - np.array([row['grasp_position'] for row in trace]), axis=1)
    first_move = np.flatnonzero(commands != 0)
    first_move = int(first_move[0]) if len(first_move) else None
    object_joint = model.joint('payload_free')
    object_qadr = int(object_joint.qposadr[0])
    object_qpos = poses[:, object_qadr:object_qadr+3]
    # mj_step integrates qpos after evaluating xpos. The original logger retained
    # that last-substep (2.5 ms) lag; check it explicitly rather than calling the
    # two streams identical. Linear interpolation is approximate under contact.
    lag_fraction = model.opt.timestep / np.diff(times)[:, None]
    expected_logged_object = object_qpos[1:] - np.diff(object_qpos, axis=0) * lag_fraction
    object_lag_residual = np.linalg.norm(expected_logged_object - objects[1:], axis=1)
    checks = dict(
        reached_complete=bool(np.any(phases == 'COMPLETE')),
        no_reported_failure=not bool(np.any(np.isin(phases, ['REACH_TIMEOUT', 'GRASP_FAILED', 'OBJECT_DROPPED']))),
        no_fall_by_orientation_and_pelvis_clearance=bool(max(r['gravity_z'] for r in trace) < -.3 and np.min(roots[:, 2] - ground) > .3),
        payload_has_free_joint=bool(object_joint.type[0] == mujoco.mjtJoint.mjJNT_FREE),
        only_finger_joint_equalities=bool(model.neq == 10 and np.all(model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
            and all('coupling' in model.equality(i).name for i in range(model.neq))),
        no_mocap_bodies=bool(model.nmocap == 0),
        arm_servo_disabled=not config['args']['arm_servo'],
        lifted_before_forward_command=bool(first_move is not None and objects[first_move, 2] - objects[0, 2] > .09),
        hand_contact_every_carry_sample=bool(np.any(carry) and np.all(hand_contact[carry])),
        no_other_object_support_during_carry=bool(np.any(carry) and not np.any(nonhand_support[carry])),
        object_remains_near_hand=bool(np.any(carry) and np.max(separation[carry]) < .09),
        both_feet_on_landing_final_3s=bool(np.all(foot_contacts) and np.all(foot_positions[:, :, :2] > bounds[0, :2])
            and np.all(foot_positions[:, :, :2] < bounds[1, :2])),
        final_root_position_range_below_2cm=bool(np.max(np.ptp(roots[final], axis=0)) < .02),
        root_trace_matches_saved_poses=bool(np.allclose(poses[:, :3], roots, atol=1e-10)),
        object_trace_consistent_with_last_substep_lag=bool(np.max(object_lag_residual) < .001),
    )
    return dict(
        success=all(checks.values()), checks=checks, source_run=str(run.resolve()),
        scope='One tuned, fixed-scene demonstration; not a randomized success-rate evaluation.',
        contact_scope='Geometric contacts sampled at 50 Hz; no substep force history was recorded.',
        duration_s=float(times[-1]), samples=len(trace), phase_transitions=transitions,
        first_forward_command_s=None if first_move is None else float(times[first_move]),
        lift_at_first_forward_command_m=None if first_move is None else float(objects[first_move, 2] - objects[0, 2]),
        min_pelvis_clearance_m=float(np.min(roots[:, 2] - ground)),
        max_root_tilt_deg=float(np.degrees(np.arccos(-max(r['gravity_z'] for r in trace)))),
        object_logging_lag_s=float(model.opt.timestep),
        max_logged_object_vs_qpos_distance_m=float(np.max(np.linalg.norm(objects-object_qpos, axis=1))),
        max_object_lag_interpolation_residual_m=float(np.max(object_lag_residual)),
        carry_hand_contact_fraction=float(np.mean(hand_contact[carry])) if np.any(carry) else None,
        max_carry_object_hand_distance_m=float(np.max(separation[carry])) if np.any(carry) else None,
        final_root_m=roots[-1].tolist(), final_object_m=objects[-1].tolist(),
        final_foot_positions_m=foot_positions[-1].tolist(),
        final_3s_root_range_m=np.ptp(roots[final], axis=0).tolist(),
        stairs=dict(steps=6, riser_m=config['args']['height'], tread_m=.32, ascent_m=float(bounds[1, 2])),
        object_mass_kg=float(model.body('payload').mass[0]),
        checkpoint=config['demo']['checkpoint'], arm_servo=config['args']['arm_servo'],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
