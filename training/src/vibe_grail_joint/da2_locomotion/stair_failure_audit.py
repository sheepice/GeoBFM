"""Bounded fixed-policy audit of stair failures, without parameter updates."""
import argparse
import json
from pathlib import Path


def monitor_edge(env):
    env.stair_failures.finalize_step()
    return env.stair_failures.edge_full & False


def monitor_pair(env):
    env.stair_failures.finalize_step()
    return env.stair_failures.pair_full & False


def find_pair_pose(model, source, prefix='robot/'):
    """Symmetric legs avoid confusing a gait phase's vertical separation with a miss."""
    import mujoco
    import numpy as np
    left = {i for i in range(model.ngeom) if model.geom(i).name.startswith(prefix+'left_foot')}
    right = {i for i in range(model.ngeom) if model.geom(i).name.startswith(prefix+'right_foot')}
    assert len(left) == len(right) == 7
    data = mujoco.MjData(model)
    symmetric = source.copy()
    for side in ('left', 'right'):
        for name, value in (('hip_yaw', 0.), ('hip_pitch', -.25), ('knee', .6),
                            ('ankle_pitch', -.35), ('ankle_roll', 0.)):
            symmetric[model.joint(prefix+side+'_'+name+'_joint').qposadr[0]] = value
    lj = model.joint(prefix+'left_hip_roll_joint').qposadr[0]
    rj = model.joint(prefix+'right_hip_roll_joint').qposadr[0]
    for angle in np.linspace(.05, .5, 91):
        data.qpos[:] = symmetric
        data.qpos[lj] = -angle; data.qpos[rj] = angle
        data.qvel[:] = 0.
        mujoco.mj_forward(model, data)
        pairs = [(int(x.geom[0]), int(x.geom[1]), float(x.dist)) for x in data.contact
                 if ((int(x.geom[0]) in left and int(x.geom[1]) in right) or
                     (int(x.geom[1]) in left and int(x.geom[0]) in right))]
        if pairs and min(p[2] for p in pairs) < -.006:
            return data.qpos.copy(), pairs, float(angle)
    raise RuntimeError('Symmetric legs produced no actual left-right foot contact')


def actual_pair_probe(raw):
    """Find a real foot-foot contact in compiled MuJoCo, then exercise Warp sensors."""
    import mujoco
    import numpy as np
    import torch
    c = raw.command_manager.get_term('motion')
    obs = raw.stair_failures
    target = c.ppo_online & (c.tile_data['kind'][c.tile] != 3)
    world = int(target.nonzero()[0])
    local = int((obs.ids == world).nonzero()[0])
    model = raw.sim.mj_model
    source = raw.sim.data.qpos[world].cpu().numpy().copy()
    found = find_pair_pose(model, source)
    raw.sim.data.qpos[world].copy_(torch.as_tensor(found[0], device=raw.device))
    raw.sim.data.qvel[world].zero_()
    raw.sim.data.qvel[world, model.joint('robot/left_hip_roll_joint').dofadr[0]] = -1.
    raw.sim.data.qvel[world, model.joint('robot/right_hip_roll_joint').dofadr[0]] = 1.
    raw.sim.forward()
    obs.age.fill_(1.)
    obs.begin_step()
    probe_samples = []
    for _ in range(raw.cfg.decimation):
        raw.sim.step(); raw.scene.update(dt=raw.physics_dt)
        obs.observe_substep()
        contact = raw.scene['stairs_left_right_foot_contact'].data
        probe_samples.append(dict(found=float(contact.found[world].sum()),
                                  peak_n=float(contact.force[world].norm(dim=-1).max())))
    obs.finalize_step()
    peak = float(obs.pair_peak[local])
    if not bool(obs.pair_full[world]):
        raise RuntimeError(f'Actual Warp foot collision was not detected; peak={peak}; samples={probe_samples}; native={found[1]}')
    return dict(world=world, hip_roll_magnitude=found[2], native_pairs=found[1],
                detected_peak_n=peak, true_failure_detected=True,
                primary_names=list(raw.scene['stairs_left_right_foot_contact'].primary_names), samples=probe_samples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True); p.add_argument('--num-envs', type=int, default=320)
    p.add_argument('--steps', type=int, default=2500)
    args = p.parse_args()
    from ..gpu_guard import verify_guarded_worker
    verify_guarded_worker()
    import torch
    from mjlab.rl import RslRlVecEnvWrapper
    from da2_vibe.paths import load_settings
    from ..da2_flat_repair.task import env_config, QualityStairsEnv
    from .evaluate import load_actor
    settings = load_settings(args.config)
    if not settings.get('stair_failures') or args.steps > 3000 or args.num_envs > 1000:
        raise ValueError('Bounded audit requires explicit stair rules, <=3000 steps, <=1000 worlds')
    torch.set_num_threads(2); torch.manual_seed(settings['seed'])
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    cfg = env_config(settings['bank_root'] + '/rank_0', settings, args.num_envs)
    cfg.terminations['stair_edge_support'].func = monitor_edge
    cfg.terminations['left_right_foot_collision'].func = monitor_pair
    success = cfg.terminations['online_success']
    success.func = success.params['base_func']; success.params = success.params['base_params']
    env = RslRlVecEnvWrapper(QualityStairsEnv(cfg, device='cuda:0', settings=settings))
    raw = env.unwrapped
    try:
        raw.policy_update_count = settings['stair_failures']['start_update'] + settings['stair_failures']['adaptation_updates']
        # PolicyUpdateCounter recomputes this from the absolute control counter.
        # Setting only the cached update would silently revert strictness next step.
        raw.common_step_counter = raw.policy_update_count * 24
        env.reset()
        probe = actual_pair_probe(raw)
        print('AUDIT actual contact probe:', json.dumps(probe), flush=True)
        obs, _ = env.reset()
        detector = raw.stair_failures
        # reset() does not erase a just-finished snapshot; the first sampled step does.
        actor, iteration = load_actor(settings, obs, args.checkpoint, raw)
        totals = dict(steps=0, touchdowns=0, edge_steps=0, pair_steps=0,
                      episodes=0, initial_pair_events=0, candidate_events=0)
        examples = []
        previous_edge = torch.zeros(len(detector.ids), device=raw.device, dtype=torch.bool)
        previous_pair = previous_edge.clone()
        kind_counts = {k: dict(steps=0, edge_steps=0, pair_steps=0) for k in ('up', 'down', 'up_down', 'flat')}
        with torch.inference_mode():
            for step in range(args.steps):
                obs, _, done, _ = env.step(actor(obs, stochastic_output=False))
                snap = detector.snapshot
                edge, pair = snap['raw_edge'], snap['raw_pair']
                kinds = raw.command_manager.get_term('motion').tile_data['kind'][snap['tile']]
                totals['steps'] += len(detector.ids)
                totals['touchdowns'] += int(detector.touchdowns.sum())
                totals['edge_steps'] += int(edge.sum()); totals['pair_steps'] += int(pair.sum())
                totals['episodes'] += int(done[detector.ids].sum())
                totals['initial_pair_events'] += int(((snap['pair_peak'] >= settings['stair_failures']['pair_peak_n']) & (snap['age'] < .2)).sum())
                for k, name in enumerate(kind_counts):
                    mask = kinds == k
                    kind_counts[name]['steps'] += int(mask.sum())
                    kind_counts[name]['edge_steps'] += int((edge & mask).sum())
                    kind_counts[name]['pair_steps'] += int((pair & mask).sum())
                candidates = ((edge & ~previous_edge) | (pair & ~previous_pair)).nonzero().flatten()
                previous_edge, previous_pair = edge.clone(), pair.clone()
                previous_edge[done[detector.ids].bool()] = False
                previous_pair[done[detector.ids].bool()] = False
                totals['candidate_events'] += len(candidates)
                for local in candidates[:max(0, min(8, 200 - len(examples)))].tolist():
                    examples.append(dict(step=step, world=int(detector.ids[local]),
                        edge=bool(edge[local]), pair=bool(pair[local]), tile=int(snap['tile'][local]),
                        margin=snap['margin'][local].cpu().tolist(), load=snap['load'][local].cpu().tolist(),
                        pair_peak_n=float(snap['pair_peak'][local]), qpos=snap['qpos'][local].cpu().tolist(),
                        root=snap['root'][local].cpu().tolist(), threshold=snap['threshold']))
                if step % 250 == 0:
                    print('AUDIT', step, json.dumps(totals), flush=True)
        if abs(detector.threshold - settings['stair_failures']['final_margin_m']) > 1e-8:
            raise RuntimeError('Fixed-policy audit silently left the final strict threshold')
        report = dict(checkpoint=args.checkpoint, checkpoint_iteration=iteration, no_learning=True,
                      actual_contact_probe=probe, totals=totals, by_kind=kind_counts,
                      threshold=detector.threshold, settings=settings, num_examples=len(examples))
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        (out/'events.json').write_text(json.dumps(examples, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k != 'settings'}, indent=2), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
