"""Fixed paired first-episode trials for ordinary cylinder stepping stones."""
import argparse
import copy
from dataclasses import dataclass, fields
import hashlib
from itertools import product
from pathlib import Path

import torch
from mjlab.rl import RslRlVecEnvWrapper
from orcs.core.robustness import strip_domain
from rsl_rl.utils import resolve_callable

from da2_vibe.paths import load_checkpoint, load_settings, profile
from ..catalog import write_json
from ..common.cuda_graph import build_sensor_env
from ..gpu_guard import verify_guarded_worker
from ..stairs_ab.evaluation_metrics import add_foot_pair_sensor, FootPairObserver
from .command import StoneCommand, StoneCommandCfg
from .hard_evaluation_terrain import CONDITIONS, hard_tiles
from .task import agent_config, env_config, StoneEnv, succeeded


class PairedCommand(StoneCommand):
    def _sample_init_frame(self, ids):
        if not hasattr(self, 'case_table'):
            self.case_table = torch.tensor(self.cfg.cases, device=self.device)
        tile_index, phase, offset = self.case_table[ids].T
        self.tile[ids] = tile_index.long()
        self.stream[ids] = 1
        self.phase[ids] = phase.long()
        self.speed[ids] = self.cfg.fixed_speed
        chosen = self.replay_clips + self.stream[ids]
        frames = self.motion.clip_offsets[chosen] + self.pool.cursor - self.phase[ids]
        spawn = self.tile_origins[self.tile[ids]].clone()
        spawn[:, 0] += offset
        spawn[:, :2] -= self.motion.body_pos_w[frames, 0, :2]
        self._env.scene.env_origins[ids] = spawn
        return chosen, frames


@dataclass(kw_only=True)
class PairedCfg(StoneCommandCfg):
    cases: tuple

    def build(self, env):
        return PairedCommand(self, env)


def stable_exit(env):
    succeeded(env)
    command = env.command_manager.get_term('motion')
    command.terminal_success = command.success_hold >= 50
    return command.terminal_success


class EpisodeObserver:
    """Take each measurement before the simulator resets completed worlds."""
    def __init__(self, env):
        self.env = env
        self.command = env.command_manager.get_term('motion')
        self.pair = FootPairObserver(env)
        self.active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        self.peaks = torch.zeros((env.num_envs, 3), device=env.device)
        self.rows = {}
        original_substep = env.metrics_manager.compute_substep
        original_compute = env.metrics_manager.compute

        def substep():
            original_substep()
            self.pair.observe_substep()

        def compute():
            original_compute()
            self.observe()

        env.metrics_manager.compute_substep = substep
        env.metrics_manager.compute = compute

    def observe(self):
        env = self.env
        g = env.guidance
        data = g.quality.step_data
        if (self.active & g.quality.nonfinite_contact).any():
            raise ValueError('Nonfinite active contact')
        current = torch.stack((data['foot_riser_force_n'].amax(-1),
                               self.pair.peak, data['shin_force_n'].amax(-1)), dim=1)
        self.peaks[self.active] = torch.maximum(self.peaks, current)[self.active]
        ended = self.active & env.reset_buf
        term = env.termination_manager.get_term
        for i in ended.nonzero().flatten().tolist():
            tile_index, phase, offset = self.command.cfg.cases[i]
            tile = self.command.cfg.random_tiles[tile_index]
            floor = bool(g.terminal_floor[i])
            failure = bool(term('stone_failure')[i])
            success = bool(term('stone_success')[i]) and not failure
            if floor:
                outcome = 'floor'
            elif failure:
                outcome = 'fall_or_out_of_bounds'
            elif success:
                outcome = 'success'
            elif term('stone_stall')[i]:
                outcome = 'stall'
            elif term('time_out')[i]:
                outcome = 'timeout'
            else:
                raise RuntimeError('Unclassified terminal state')
            wall, pair, shin = self.peaks[i].tolist()
            self.rows[i] = dict(case=i, level=tile['level'], variant=tile['variant'], phase=phase,
                offset_m=offset, seconds=round(int(env.episode_length_buf[i]) * env.step_dt, 4),
                outcome=outcome, completed_pillars=int(g.sequence.progress[i]),
                total_pillars=int(self.command.tile_data['n'][self.command.tile[i]]),
                wall_peak_n=wall, pair_peak_n=pair, shin_peak_n=shin,
                clean_success=success and max(wall, pair, shin) <= 15)
            if 'terrain_seed' in tile:
                self.rows[i]['terrain_seed'] = tile['terrain_seed']
        self.active &= ~ended


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    settings = load_settings(args.config)
    settings['seed'] = args.seed
    settings['stones']['local_coordinates'] = True
    tiles = hard_tiles(args.hard_condition) if args.hard_condition else None
    n_tiles = len(tiles) if tiles else settings['stones']['levels'] * settings['profile_variants']
    cases = tuple(product(range(n_tiles), (0, 5, 10, 15), (-.1, .1)))
    n = len(cases)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    cfg = env_config(settings['schema_bank'], settings, n)
    old = cfg.commands['motion']
    if tiles:
        old.random_tiles = tiles
        cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs'].random_tiles = tiles
    cfg.commands['motion'] = cmd = PairedCfg(cases=cases, **{f.name: getattr(old, f.name) for f in fields(old)})
    cmd.pose_range = {}
    cmd.velocity_range = {}
    cmd.joint_position_range = (0., 0.)
    cfg.terminations['stone_success'].func = stable_exit
    add_foot_pair_sensor(cfg)
    strip_domain(cfg)
    cfg.events.pop('policy_update_counter', None)
    for group in cfg.observations.values():
        group.enable_corruption = False
        for term in group.terms.values():
            term.noise = None
    raw = build_sensor_env(StoneEnv, cfg, device='cuda:0', settings=settings,
        disable_graph_capture=profile().get('runtime', {}).get('disable_cuda_graph_capture', False))
    env = RslRlVecEnvWrapper(raw)
    try:
        obs = env.get_observations()
        c = raw.command_manager.get_term('motion')
        if c.phase.tolist() != [p for _,p,_ in cases] or c.tile.tolist() != [t for t,_,_ in cases]:
            raise ValueError('Initial cases do not match the requested matrix')
        robot = c.robot.data
        initial = torch.cat((robot.root_link_pos_w, robot.root_link_quat_w, robot.root_link_lin_vel_w,
                  robot.root_link_ang_vel_w, robot.joint_pos, robot.joint_vel), dim=-1)
        write_json(out / 'initial.json', dict(states=initial.cpu().tolist(), tiles=cmd.random_tiles))
        agent = agent_config(settings['schema_bank'], settings, n)
        kwargs = copy.deepcopy(agent['actor'])
        actor = resolve_callable(kwargs.pop('class_name'))(obs=obs, obs_groups=agent['obs_groups'],
                       obs_set='actor', output_dim=29, **kwargs).to('cuda:0').eval()
        checkpoint = load_checkpoint(args.checkpoint, map_location='cpu', weights_only=False)
        actor.load_state_dict(checkpoint['actor_state_dict'], strict=True)
        observer = EpisodeObserver(raw)
        with torch.inference_mode():
            for step in range(raw.max_episode_length):
                observer.pair.begin_step()
                obs, _, _, _ = env.step(actor(obs, stochastic_output=False))
                if (step + 1) % 100 == 0:
                    print(f'Control step {step+1}, finished {len(observer.rows)}/{n}', flush=True)
                if len(observer.rows) == n:
                    break
        if len(observer.rows) != n:
            raise RuntimeError(f'Incomplete trial matrix: {len(observer.rows)}/{n}')
        write_json(out / 'report.json', dict(seed=args.seed, checkpoint=str(args.checkpoint),
                   sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   iteration=int(checkpoint['iter']), method=settings['stones_comparison']['method'],
                   control_dt=raw.step_dt, physics_dt=raw.physics_dt,
                   geometry_condition=args.hard_condition, episodes=[observer.rows[i] for i in range(n)]))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--hard-condition', choices=CONDITIONS)
    args = parser.parse_args()
    verify_guarded_worker()
    run(args)


if __name__ == '__main__':
    main()
