"""Fixed-seed first-episode checks of the actual policy and contact sequence."""
from da2_vibe.paths import load_checkpoint
from da2_vibe.paths import load_settings
import argparse
import copy
import json
from dataclasses import dataclass, fields
from pathlib import Path
from ..gpu_guard import verify_guarded_worker


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--level', type=int, default=0)
    p.add_argument('--seed', type=int, default=261103)
    p.add_argument('--video', action='store_true')
    args = p.parse_args()
    verify_guarded_worker()
    import mjlab
    import numpy as np
    import torch
    from mjlab.rl import RslRlVecEnvWrapper
    from orcs.core.robustness import strip_domain
    from rsl_rl.utils import resolve_callable
    from .task import env_config, agent_config, StoneEnv
    from .command import StoneCommand, StoneCommandCfg
    from .initialization import load_stage1
    from ..common.review_camera import add_review_camera
    from ..catalog import write_json
    settings = load_settings(args.config)
    settings['seed'] = args.seed
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    n = 8

    class FixedCommand(StoneCommand):
        def _sample_init_frame(self, ids):
            chosen, frames = super()._sample_init_frame(ids)
            self.tile[ids] = args.level * self.cfg.profile_variants
            self.phase[ids] = (ids % 4) * 5
            frames = self.motion.clip_offsets[chosen] + self.pool.cursor - self.phase[ids]
            origin = self.tile_origins[self.tile[ids]].clone()
            origin[:, 0] += torch.where(ids < 4, -.1, .1)
            origin[:, :2] -= self.motion.body_pos_w[frames, 0, :2]
            self._env.scene.env_origins[ids] = origin
            return chosen, frames

    @dataclass(kw_only=True)
    class FixedCfg(StoneCommandCfg):
        def build(self, env):
            return FixedCommand(self, env)

    cfg = env_config(settings['schema_bank'], settings, n)
    old = cfg.commands['motion']
    cfg.commands['motion'] = FixedCfg(**{f.name: getattr(old, f.name) for f in fields(FixedCfg)})
    tile = cfg.commands['motion'].random_tiles[args.level * settings['profile_variants']]
    if args.video:
        add_review_camera(cfg, dict(tile, base=tile['height']))
    strip_domain(cfg)
    cfg.events.pop('policy_update_counter', None)
    for group in cfg.observations.values():
        group.enable_corruption = False
        for term in group.terms.values():
            term.noise = None
    raw = StoneEnv(cfg, device='cuda:0', settings=settings)
    env = RslRlVecEnvWrapper(raw)
    obs = env.get_observations()
    agent = agent_config(settings['schema_bank'], settings, n)
    actor_cfg = copy.deepcopy(agent['actor'])
    cls = resolve_callable(actor_cfg.pop('class_name'))
    actor = cls(obs=obs, obs_groups=agent['obs_groups'], obs_set='actor', output_dim=29, **actor_cfg).to('cuda:0')
    if args.checkpoint:
        state = load_checkpoint(args.checkpoint, map_location='cpu', weights_only=False)
        actor.load_state_dict(state['actor_state_dict'], strict=True)
        model = dict(checkpoint=args.checkpoint, iteration=state['iter'])
    else:
        model = load_stage1(actor, settings['stage1_checkpoint'])
    actor.eval()
    c = raw.command_manager.get_term('motion')
    initial = dict(root=c.robot.data.root_link_pos_w.cpu().tolist(),
                   joints=c.robot_joint_pos.cpu().tolist(), phase=c.phase.cpu().tolist())
    active = torch.ones(n, device='cuda:0', dtype=torch.bool)
    max_pillars = torch.zeros(n, device='cuda:0')
    foot_impulse = torch.zeros(n, device='cuda:0')
    shin_impulse = foot_impulse.clone()
    results = [None] * n
    video = None
    frames = 0
    if args.video:
        import imageio.v2 as iio
        video = iio.get_writer(out / 'rollout.mp4', fps=25, codec='libx264', quality=7)
    try:
        with torch.inference_mode():
            for step in range(1750):
                recording = bool(active[0])
                obs, _, done, _ = env.step(actor(obs, stochastic_output=False))
                g = raw.guidance
                max_pillars = torch.maximum(max_pillars, g.terminal_progress * tile['n'] * active)
                foot_impulse += g.quality.step_data['foot_riser_impulse_ns'].sum(-1) * active
                shin_impulse += g.quality.step_data['shin_impulse_ns'].sum(-1) * active
                if video is not None and recording and step % 2 == 0:
                    video.append_data(raw.scene['review_camera'].data.rgb[0].cpu().numpy())
                    frames += 1
                ended = active & done.bool()
                for i in ended.nonzero().flatten().tolist():
                    results[i] = dict(case=i, seconds=(step + 1) * .02,
                        strict_success=bool(c.terminal_success[i]), fall=bool(c.terminal_fall[i]),
                        floor_contact=bool(g.terminal_floor[i]), alternating_pillars=int(max_pillars[i]),
                        total_pillars=tile['n'], foot_impulse_ns=float(foot_impulse[i]), shin_impulse_ns=float(shin_impulse[i]))
                active &= ~done.bool()
                if not active.any():
                    break
        for i in active.nonzero().flatten().tolist():
            results[i] = dict(case=i, seconds=35., strict_success=False, timed_out=True,
                              alternating_pillars=int(max_pillars[i]), total_pillars=tile['n'])
        write_json(out / 'report.json', dict(model=model, level=args.level, seed=args.seed, tile=tile,
            initial=initial, cases=results, strict_successes=sum(r['strict_success'] for r in results),
            cases_count=n, video_frames=frames, protocol='8 fixed initial states; first episode only; actual policy output; no action mixing'))
    finally:
        if video:
            video.close()
        env.close()


if __name__ == '__main__':
    main()
