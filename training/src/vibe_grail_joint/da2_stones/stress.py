"""Bounded checkpoint rollout under training randomization and stochastic actions.

This numerical regression check retains automatic resets, both mirror layouts,
and flat mixtures. It is not a policy success-rate benchmark or a PPO update.
"""
import argparse
import copy
import time
from pathlib import Path
from da2_vibe.paths import load_checkpoint, load_settings
from ..gpu_guard import verify_guarded_worker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--num-envs', type=int, default=1280)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--level', type=int, default=6)
    p.add_argument('--seed', type=int, default=269026)
    p.add_argument('--domain-ramp-complete', action='store_true')
    args = p.parse_args()
    verify_guarded_worker()
    import mjlab
    import torch
    from mjlab.rl import RslRlVecEnvWrapper
    from rsl_rl.utils import resolve_callable
    from .task import env_config, agent_config, StoneEnv
    from ..catalog import write_json
    if args.num_envs % 20 or not 1 <= args.steps <= 10000:
        raise ValueError('Use a multiple of 20 worlds and 1..10000 steps')
    settings = load_settings(args.config)
    if not 0 <= args.level < settings['stones']['levels']:
        raise ValueError('Invalid curriculum level')
    settings['seed'] = args.seed
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    record = dict(status='initializing', settings=settings, checkpoint=args.checkpoint,
                  num_envs=args.num_envs, steps=args.steps, frontier=args.level, seed=args.seed,
                  protocol='Training randomization; stochastic checkpoint actions; automatic resets; no optimizer updates')
    write_json(out / 'status.json', record)
    cfg = env_config(settings['schema_bank'], settings, args.num_envs)
    if args.domain_ramp_complete:
        # No PPO updates occur in this probe. Do not let the rollout clock
        # replace the fixed post-ramp update with step_count / 24.
        cfg.events.pop('policy_update_counter',None)
    raw = StoneEnv(cfg, device='cuda:0', settings=settings)
    if args.domain_ramp_complete:
        domain=settings['appearance_randomization']
        raw.policy_update_count=domain['start_update']+domain['ramp_updates']
        record['domain_ramp_complete']=True
    raw.failure_snapshot_dir = out
    env = RslRlVecEnvWrapper(raw)
    c = raw.command_manager.get_term('motion')
    c.frontier = args.level
    env.reset()
    obs = env.get_observations()
    agent = agent_config(settings['schema_bank'], settings, args.num_envs)
    actor_cfg = copy.deepcopy(agent['actor'])
    cls = resolve_callable(actor_cfg.pop('class_name'))
    actor = cls(obs=obs, obs_groups=agent['obs_groups'], obs_set='actor', output_dim=29,
                **actor_cfg).to('cuda:0')
    state = load_checkpoint(args.checkpoint, map_location='cpu', weights_only=False)
    actor.load_state_dict(state['actor_state_dict'], strict=True)
    actor.eval()
    try:
        with torch.inference_mode():
            for step in range(args.steps):
                obs, reward, _, _ = env.step(actor(obs, stochastic_output=True))
                if not torch.isfinite(reward).all():
                    raise RuntimeError('Nonfinite total reward')
                if (step + 1) % 100 == 0:
                    record.update(status='running', completed_steps=step+1,
                                  transitions=(step+1)*args.num_envs, elapsed_s=time.monotonic()-start,
                                  pillar_episodes=raw.completed_episodes, flat_episodes=raw.flat_episodes)
                    write_json(out / 'status.json', record)
                    print({k: record[k] for k in ('completed_steps','transitions','elapsed_s')}, flush=True)
        if args.domain_ramp_complete:
            expected=domain['start_update']+domain['ramp_updates']
            assert raw.policy_update_count==expected
            record['policy_update_count']=raw.policy_update_count
        record.update(status='passed', finite=True, elapsed_s=time.monotonic()-start)
    except BaseException as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        write_json(out / 'status.json', record)
        env.close()


if __name__ == '__main__':
    main()
