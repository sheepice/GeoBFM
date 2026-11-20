"""Actual simulator ray hits supervise geometry; privileged data never enters actor inputs."""
import torch
from tensordict import TensorDict
from ..velocity.dagger_env import DaggerVecEnv


class GeometryDaggerEnv(DaggerVecEnv):
    @property
    def quality_steps(self):
        return self.env.quality_steps

    @quality_steps.setter
    def quality_steps(self, value):
        self.env.quality_steps = value


@torch.no_grad()
def terrain_targets(raw, ids, height_scale_m=1.):
    sensor = raw.scene['terrain_scan']
    offsets, _ = sensor.cfg.pattern.generate_rays(None, raw.device)
    front = (offsets[:, 0] >= -1e-6).nonzero().flatten()
    if len(front) != 99:
        raise ValueError(f'Expected 99 forward rays, found {len(front)}')
    data = sensor.data
    hits = data.hit_pos_w[ids][:, front]
    valid = (data.distances[ids][:, front] >= 0) & torch.isfinite(hits).all(-1)
    height = (hits[..., 2] - data.pos_w[ids, None, 2]) / height_scale_m
    height = torch.where(valid, height, 0.)
    return height, valid


@torch.no_grad()
def audit_terrain_labels(raw):
    """Compare actual mesh ray hits to independent analytic generated-stair heights."""
    command = raw.command_manager.get_term('motion')
    ids = command.ppo_online.nonzero().flatten()[:256]
    sensor = raw.scene['terrain_scan']
    data = sensor.data
    hits = data.hit_pos_w[ids]
    predicted = command.ground_height(hits[..., :2], ids)
    valid = (data.distances[ids] >= 0) & torch.isfinite(hits).all(-1)
    error = (predicted-hits[..., 2]).abs()
    # At an exact vertical edge, float32 ray/primitive implementations can
    # choose opposite sides. Accept only a height present within 2 micrometers;
    # centimeter misalignment away from an edge must still fail this audit.
    ambiguous = valid & (error > .002)
    boundary_count = 0
    if ambiguous.any():
        for dx,dy in ((2e-6,0.),(-2e-6,0.),(0.,2e-6),(0.,-2e-6)):
            offset=hits.new_tensor([dx,dy])
            neighbor=command.ground_height(hits[..., :2]+offset,ids)
            error=torch.minimum(error,(neighbor-hits[...,2]).abs())
        boundary_count=int((ambiguous & (error <= .002)).sum())
    error=error[valid]
    if not len(error) or error.max() > .002:
        raise RuntimeError(f'Terrain ray-label geometry mismatch: {float(error.max()) if len(error) else None}')
    return dict(checked_rays=len(error), max_error_m=float(error.max()),
                boundary_rays=boundary_count, boundary_xy_tolerance_m=2e-6,
                paired_grail_uses_actual_mesh_ray_hits=True, actor_receives_labels=False)


class GeometryCollector:
    def __init__(self, wrapped, settings, rank, enabled=True):
        self.env, self.settings = wrapped, dict(settings)
        self.enabled = enabled
        self.pending, self.replay = [], None
        self.last_step, self.total_samples = -1, 0
        self.rng = torch.Generator().manual_seed(291031 + rank)
        self.last_metrics = {}

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def posture_steps(self):
        return self.env.posture_steps

    @posture_steps.setter
    def posture_steps(self, value):
        self.env.posture_steps = value

    @property
    def quality_steps(self):
        return self.env.quality_steps

    @quality_steps.setter
    def quality_steps(self, value):
        self.env.quality_steps = value

    @torch.no_grad()
    def capture(self, obs):
        if not self.enabled:
            return
        step = int(self.unwrapped.common_step_counter)
        if step == self.last_step or step % self.settings['sample_interval_steps']:
            return
        self.last_step = step
        command = self.unwrapped.command_manager.get_term('motion')
        available = command.ppo_mask.nonzero().flatten()
        order = torch.randperm(len(available), generator=self.rng)[:self.settings['sample_worlds']].to(available.device)
        ids = available[order]
        height, valid = terrain_targets(self.unwrapped, ids, self.settings['height_scale_m'])
        with torch.inference_mode(False):
            batch = TensorDict(dict(
                obs=obs[ids].select('kv_tokens', 'q_proprio', 'velocity_command').detach().cpu().clone(),
                height=height.cpu().clone(), valid=valid.cpu().clone()), batch_size=[len(ids)])
        self.pending.append(batch)

    def get_observations(self):
        obs = self.env.get_observations()
        self.capture(obs)
        return obs

    def reset(self):
        self.pending.clear()
        self.last_step = -1
        obs, extras = self.env.reset()
        self.capture(obs)
        return obs, extras

    def step(self, actions):
        obs, reward, done, extras = self.env.step(actions)
        self.capture(obs)
        return obs, reward, done, extras

    def prepare(self):
        if not self.pending:
            raise RuntimeError('No fresh terrain supervision samples')
        fresh = torch.cat(self.pending)
        self.pending.clear()
        self.total_samples += len(fresh)
        self.replay = fresh if self.replay is None else torch.cat((self.replay, fresh))
        self.replay = self.replay[-self.settings['buffer_capacity']:].clone()
        self.last_metrics = dict(fresh_samples=len(fresh), replay_samples=len(self.replay),
                                 total_samples=self.total_samples, valid_fraction=float(fresh['valid'].float().mean()))

    def sample(self, count, device):
        ids = torch.randint(len(self.replay), (count,), generator=self.rng)
        return self.replay[ids].to(device).clone()

    def discard_sample(self, count):
        """Keep capture/replay RNG matched when geometry supervision is ablated."""
        torch.randint(len(self.replay), (count,), generator=self.rng)

    def state_dict(self):
        return dict(version=1, settings=self.settings, rng=self.rng.get_state(),
                    total_samples=self.total_samples, replay=self.replay)

    def load_state_dict(self, state):
        if state['version'] != 1 or state['settings'] != self.settings:
            raise ValueError('Geometry replay contract mismatch')
        self.rng.set_state(state['rng'])
        self.total_samples, self.replay = state['total_samples'], state['replay']
