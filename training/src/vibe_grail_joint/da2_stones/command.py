"""Online references only; course geometry never enters actor observations."""
from dataclasses import dataclass
import torch
from ..velocity.command import VelocityMotionCommand
from ..velocity.dagger_task import DaggerMotionCommandCfg
from .terrain import PrimitiveHeights


class StoneCommand(VelocityMotionCommand):
    def _init_task(self):
        super()._init_task()
        self.online[:] = True
        self.paired[:] = False
        self.flat[:] = False
        self.ppo_online = self.online.clone()
        self.ppo_mask = self.online.clone()
        self.dagger_mask = self.flat.clone()
        self.dagger_planner = self.flat.clone()
        self.online_ids = self.online.nonzero().flatten()
        self.metrics['terrain_fraction'] = self.online.float()
        self.primitives = PrimitiveHeights(self.cfg.random_tiles, self.device)
        self.frontier = 0
        self.curriculum_counts = torch.zeros(self.cfg.stone_levels, 3, device=self.device)
        self.recent_counts = torch.zeros(4, self.cfg.stone_levels, 3, device=self.device)
        self.last_promotion = 0
        self.course_start = torch.tensor([t['start'] for t in self.cfg.random_tiles], device=self.device)
        self.course_end = torch.tensor([t['course_end'] for t in self.cfg.random_tiles], device=self.device)
        self.flat_tile_start = next((i for i, t in enumerate(self.cfg.random_tiles)
                                    if t.get('family') == 'flat_training'), len(self.cfg.random_tiles))
        if self.cfg.level10:
            from .level10 import LEVEL, route_groups
            tiles = self.cfg.random_tiles
            size = self.primitives.values.shape[1]
            routes = [t.get('route_targets', list(range(t['n']))) for t in tiles]
            self.route_targets = torch.tensor([ids + [-1] * (size - len(ids)) for ids in routes], device=self.device)
            self.route_count = torch.tensor([len(ids) for ids in routes], device=self.device)
            self.route_kind = torch.tensor([t.get('route_kind', 0) for t in tiles], device=self.device)
            self.route_first_foot = torch.tensor([
                t.get('route_first_foot', (0 if t['solids'][0]['y'] > 0 else 1) if t['solids'] else -1)
                for t in tiles], device=self.device)
            self.route_exit = torch.tensor([t.get('route_exit', t['course_end']) for t in tiles], device=self.device)
            self.approach_x_scale = torch.where(self.route_kind > 0, .25, 1.)
            self.route_group_by_env = route_groups(torch.arange(self.num_envs, device=self.device))
            self.route_tiles = tuple(((self.tile_data['level'] == LEVEL) & (self.route_kind == group)).nonzero().flatten()
                                     for group in range(3))
        self.terminal_success = torch.zeros_like(self.online)
        self.terminal_fall = torch.zeros_like(self.online)
        if self.cfg.local_coordinates:
            from .local_coordinates import LocalCourseCoordinates
            self.local_frame = LocalCourseCoordinates(self._env, self.tile_origins,
                self.cfg.random_tiles, self.cfg.physical_tiles)
            self.tile_origins.zero_()

    def _reset_task(self, env_ids, clip_ids, time_steps, origins):
        if getattr(self.cfg, 'local_coordinates', False):
            self.local_frame.select(env_ids, self.tile[env_ids])
        super()._reset_task(env_ids, clip_ids, time_steps, origins)

    def ground_height(self, xy, env_ids=None):
        if not hasattr(self, 'primitives'):
            return torch.zeros(xy.shape[:-1], device=self.device)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        return self.primitives.query(xy, self.tile[ids], self.tile_origins)[0]

    def surface_ids(self, xy, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        return self.primitives.query(xy, self.tile[ids], self.tile_origins)[1]

    def _sample_init_frame(self, env_ids):
        n = len(env_ids)
        level10 = getattr(self.cfg, 'level10', None)
        maximum = min(self.frontier, 9) if level10 else self.frontier
        level = torch.randint(maximum + 1, (n,), device=self.device)
        level[torch.rand(n, device=self.device) < .7] = maximum
        variant = torch.randint(self.cfg.profile_variants, (n,), device=self.device)
        if self.cfg.layout_settings:
            s=self.cfg.layout_settings;copies=s['layouts_per_variant']
            base=self.cfg.profile_variants//copies
            progress=min(1.,max(0.,(int(getattr(self._env,'policy_update_count',s['start_update']))-s['start_update'])/s['ramp_updates']))
            nominal=torch.rand(n,device=self.device)<(.7-.45*progress)
            copy=torch.randint(copies//4,copies,(n,),device=self.device)
            copy[nominal]=torch.randint(copies//4,(int(nominal.sum()),),device=self.device)
            variant=copy*base+torch.randint(base,(n,),device=self.device)
        self.tile[env_ids] = level * self.cfg.profile_variants + variant
        if level10 and self.frontier == 10:
            pillar_ids = env_ids[self.pillar_task[env_ids]]
            for group, candidate in enumerate(self.route_tiles):
                ids = pillar_ids[self.route_group_by_env[pillar_ids] == group]
                self.tile[ids] = candidate[torch.randint(len(candidate), (len(ids),), device=self.device)]
        self.stream[env_ids] = 1
        self.phase[env_ids] = torch.randint(0, 20, (n,), device=self.device)
        self.speed[env_ids] = self.cfg.fixed_speed
        chosen = self.replay_clips + self.stream[env_ids]
        frames = self.motion.clip_offsets[chosen] + self.pool.cursor - self.phase[env_ids]
        spawn = self.tile_origins[self.tile[env_ids]].clone()
        spawn[:, 0] += torch.empty(n, device=self.device).uniform_(*self.cfg.approach_spawn_x)
        spawn[:, 1] += torch.empty(n, device=self.device).uniform_(-.04, .04)
        spawn[:, 2] = 0.
        spawn[:, :2] -= self.motion.body_pos_w[frames, 0, :2]
        self._env.scene.env_origins[env_ids] = spawn
        return chosen, frames

    def _update_command(self, env_ids=None):
        super()._update_command(env_ids)
        self.speed.fill_(self.cfg.fixed_speed)

    def update_curriculum(self, update):
        if update % 50:
            return {}
        counts = self.curriculum_counts.clone()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts)
        self.curriculum_counts.zero_()
        self.recent_counts[(update // 50 - 1) % 4] = counts
        total = self.recent_counts.sum(0)[self.frontier]
        episodes, successes, progress_sum = total.tolist()
        rate = successes / max(1, episodes)
        old = self.frontier
        if episodes >= 400 and rate >= .65 and update - self.last_promotion >= 150:
            self.frontier = min(self.frontier + 1, self.cfg.stone_levels - 1)
            if self.frontier != old:
                self.last_promotion = update
        return dict(frontier=self.frontier, evaluated_level=old, recent_episodes=episodes,
                    recent_strict_success_rate=rate, recent_fraction_pillars=progress_sum / max(1, episodes))

    def curriculum_state(self):
        return dict(frontier=self.frontier, last_promotion=self.last_promotion,
                    recent_counts=self.recent_counts.cpu(), pending=self.curriculum_counts.cpu())

    def restore_curriculum(self, state):
        self.frontier = state['frontier']
        self.last_promotion = state['last_promotion']
        self.recent_counts.copy_(state['recent_counts'])
        self.curriculum_counts.copy_(state['pending'])


@dataclass(kw_only=True)
class StoneCommandCfg(DaggerMotionCommandCfg):
    stone_levels: int = 7
    fixed_speed: float = .6
    local_coordinates: bool = False
    physical_tiles: tuple = ()
    approach_spawn_x: tuple = (-.15, .15)
    layout_settings: dict | None = None
    level10: dict | None = None

    def build(self, env):
        return StoneCommand(self, env)
