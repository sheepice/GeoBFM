"""2026-11-02 review: all eight videos, terminal snapshots before reset, traces."""
from da2_vibe.paths import load_checkpoint
from da2_vibe.paths import load_settings
import argparse
import copy
import json
from dataclasses import dataclass, fields, replace
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
    p.add_argument('--variant', type=int, default=0)
    p.add_argument('--approach-extra-m', type=float, default=0.,
                   help='Extend the flat approach by translating the selected pillar course forward.')
    args = p.parse_args()
    if not 0. <= args.approach_extra_m <= 3.:
        p.error('--approach-extra-m must be between 0 and 3 m (atlas clearance)')
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
            self.tile[ids] = args.level * self.cfg.profile_variants + args.variant
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
    tile = cfg.commands['motion'].random_tiles[args.level * settings['profile_variants'] + args.variant]
    # Command geometry and the physical atlas share this tile. Keep both aligned,
    # leaving all pillar shapes, gaps, heights, and initial robot states intact.
    for key in ('start', 'goal', 'course_end'):
        tile[key] += args.approach_extra_m
    for stone in tile['solids']:
        stone['x'] += args.approach_extra_m
    for mat in tile.get('mats', []):
        mat['x'] += args.approach_extra_m
    if args.video:
        add_review_camera(cfg, dict(tile, base=tile['height']))
        cfg.scene.sensors = tuple(replace(s, width=1280, height=720) if s.name == 'review_camera' else s
                                  for s in cfg.scene.sensors)
    strip_domain(cfg)
    cfg.events.pop('policy_update_counter', None)
    for group in cfg.observations.values():
        group.enable_corruption = False
        for term in group.terms.values():
            term.noise = None
    class ObservedStoneEnv(StoneEnv):
        def __init__(self, *a, **kw):
            self.terminal_snapshots = {}
            super().__init__(*a, **kw)

        def _reset_idx(self, env_ids=None):
            if hasattr(self, 'guidance'):
                ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
                c = self.command_manager.get_term('motion')
                for i in ids[self.episode_length_buf[ids] > 0].tolist():
                    if i not in self.terminal_snapshots:
                        origin = c.tile_origins[c.tile[i]]
                        self.terminal_snapshots[i] = dict(
                            strict_success=bool(c.terminal_success[i]),
                            task_failure=bool(c.terminal_fall[i]),
                            floor_contact=bool(self.guidance.terminal_floor[i]),
                            alternating_pillars=int(self.guidance.sequence.progress[i]),
                            terminal_causes=[name for name in cfg.terminations
                                             if bool(self.termination_manager.get_term(name)[i])],
                            root_local=(c.robot.data.root_link_pos_w[i]-origin).cpu().tolist(),
                            feet_local=(c.robot_body_pos_w[i, [3,6]]-origin).cpu().tolist(),
                            gravity_z=float(c.robot.data.projected_gravity_b[i,2]))
            return super()._reset_idx(env_ids)

    raw = ObservedStoneEnv(cfg, device='cuda:0', settings=settings)
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
    videos = []
    frames = [0] * n
    trace = {k: [] for k in ('time_s','active','root_local','feet_local','progress','support')}
    foot_events = torch.zeros(n, 2, dtype=torch.long, device='cuda:0')
    shin_events = foot_events.clone()
    last_foot = torch.zeros(n, 2, dtype=torch.bool, device='cuda:0')
    last_shin = last_foot.clone()
    events = [[] for _ in range(n)]
    previous_progress = torch.zeros(n, dtype=torch.long, device='cuda:0')
    if args.video:
        import imageio.v2 as iio
        from PIL import Image, ImageDraw, ImageFont
        import matplotlib
        font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'), 21)
        videos = [iio.get_writer(out / f'case_{i:02d}.mp4', fps=25, codec='libx264', quality=7,
                  macro_block_size=1, ffmpeg_params=['-threads','1','-preset','fast','-movflags','+faststart']) for i in range(n)]
    try:
        with torch.inference_mode():
            for step in range(1750):
                # Read and record PRE-step states. Auto-reset images must never
                # appear at the end of an episode's video or in terminal data.
                origin = c.tile_origins[c.tile]
                trace['time_s'].append(step * .02)
                for key,value in dict(active=active,root_local=c.robot.data.root_link_pos_w-origin,
                        feet_local=c.robot_body_pos_w[:,[3,6]]-origin[:,None],
                        progress=raw.guidance.sequence.progress,support=raw.guidance.quality.sensor_support).items():
                    trace[key].append(value.cpu().numpy().copy())
                if videos and step % 2 == 0:
                    rgb = raw.scene['review_camera'].data.rgb.cpu().numpy()
                    first_person = raw.scene['head_cam'].data.rgb.cpu().numpy()
                    for i in active.nonzero().flatten().tolist():
                        frame = Image.fromarray(rgb[i].copy())
                        frame.paste(Image.fromarray(first_person[i]).resize((224,224)), (1046,60))
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0,0,1280,44),fill=(20,25,30))
                        draw.text((16,9),f'Level {args.level} | Case {i} | 0.6 m/s | t={step*.02:.2f}s | Ordered landings: {int(raw.guidance.sequence.progress[i])}/{tile["n"]}',font=font,fill='white')
                        videos[i].append_data(np.asarray(frame));frames[i]+=1
                obs, _, done, _ = env.step(actor(obs, stochastic_output=False))
                g = raw.guidance
                max_pillars = torch.maximum(max_pillars, g.terminal_progress * tile['n'] * active)
                foot_impulse += g.quality.step_data['foot_riser_impulse_ns'].sum(-1) * active
                shin_impulse += g.quality.step_data['shin_impulse_ns'].sum(-1) * active
                foot_hit = g.quality.step_data['foot_riser_force_n'] > 15.
                shin_hit = g.quality.step_data['shin_force_n'] > 15.
                foot_events += (foot_hit & ~last_foot & active[:,None]).long()
                shin_events += (shin_hit & ~last_shin & active[:,None]).long()
                last_foot,last_shin = foot_hit,shin_hit
                progress = (g.terminal_progress * tile['n']).round().long()
                for i in (active & (progress > previous_progress)).nonzero().flatten().tolist():
                    events[i].append(dict(time_s=(step+1)*.02, visited=int(progress[i]),
                        foot=int(g.sequence.last_foot[i]) if not done[i] else None))
                previous_progress = torch.maximum(previous_progress,progress * active)
                ended = active & done.bool()
                for i in ended.nonzero().flatten().tolist():
                    results[i] = dict(case=i, seconds=(step + 1) * .02,
                        **raw.terminal_snapshots[i], total_pillars=tile['n'],
                        foot_impulse_ns=float(foot_impulse[i]), shin_impulse_ns=float(shin_impulse[i]),
                        foot_impact_events=int(foot_events[i].sum()), shin_impact_events=int(shin_events[i].sum()),
                        ordered_landing_events=events[i],video=f'case_{i:02d}.mp4',video_frames=frames[i])
                active &= ~done.bool()
                if not active.any():
                    break
        for i in active.nonzero().flatten().tolist():
            results[i] = dict(case=i, seconds=35., strict_success=False, timed_out=True,
                              alternating_pillars=int(max_pillars[i]), total_pillars=tile['n'])
        np.savez_compressed(out / 'trajectories.npz',**{k:np.asarray(v) for k,v in trace.items()})
        write_json(out / 'report.json', dict(model=model, level=args.level, variant=args.variant, seed=args.seed, tile=tile,
            approach_extra_m=args.approach_extra_m,
            initial=initial, cases=results, strict_successes=sum(r['strict_success'] for r in results),
            cases_count=n, video_frames=frames, protocol='8 fixed initial states; first episode only; actual policy output; no action mixing; terminal states captured before reset; video ends on last pre-terminal frame'))
    finally:
        for video in videos:
            video.close()
        env.close()


if __name__ == '__main__':
    main()
