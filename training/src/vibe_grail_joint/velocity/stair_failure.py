"""Opt-in, contact-resolved stair failures, sampled at physics frequency.

The state machine is independent of the simulator. The adapter only converts
actual contact sensors and foot poses into its inputs. No actor inputs change.
"""
import copy
import math

import torch
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply


def validate(options):
    if options.get('version') != 1 or options.get('scope') != 'online_ppo_stairs':
        raise ValueError('Unsupported stair-failure contract')
    positive = ('edge_hold_s', 'safe_hold_s', 'safe_margin_m', 'load_enter_fraction',
                'load_exit_fraction', 'minimum_load_n', 'dominance_fraction',
                'height_tolerance_m', 'unload_max_s', 'pair_peak_n', 'pair_light_n',
                'pair_hold_s', 'pair_impulse_ns', 'startup_s', 'startup_edge_distance_m')
    for key in positive:
        if not isinstance(options.get(key), (float, int)) or not 0 < options[key] < math.inf:
            raise ValueError(f'Invalid stair-failure option: {key}')
    for key in ('initial_margin_m', 'final_margin_m'):
        if not isinstance(options.get(key), (float, int)) or not math.isfinite(options[key]):
            raise ValueError(f'Invalid margin: {key}')
    for key in ('start_update', 'adaptation_updates'):
        if type(options.get(key)) is not int or options[key] < 0:
            raise ValueError(f'Invalid adaptation count: {key}')
    if not (0 < options['load_exit_fraction'] < options['load_enter_fraction'] < 1
            and .5 < options['dominance_fraction'] <= 1
            and options['initial_margin_m'] <= options['final_margin_m'] < options['safe_margin_m']
            and options['pair_light_n'] < options['pair_peak_n']):
        raise ValueError('Inconsistent stair-failure thresholds')


def validate_resume_options(previous, current, migrate=False):
    old, new = copy.deepcopy(previous), copy.deepcopy(current)
    if not isinstance(old, dict) or not isinstance(new, dict):
        raise ValueError('Missing complete experiment options')
    if migrate:
        if 'stair_failures' in old or 'stair_failures' not in new:
            raise ValueError('Stair failure migration must enable the rules exactly once')
        validate(new.pop('stair_failures'))
    if old != new:
        raise ValueError('Stair failure resume changed unrelated camera, terrain, rewards or visual options')


def add_sensors(cfg):
    sensors = list(cfg.scene.sensors)
    for side in ('left', 'right'):
        sensors.append(ContactSensorCfg(
            name=f'stairs_load_{side}',
            primary=ContactMatch(mode='geom', pattern=f'{side}_foot[1-7]_collision', entity='robot'),
            secondary=ContactMatch(mode='body', pattern='terrain'),
            fields=('found', 'force'), reduce='netforce'))
    sensors.append(ContactSensorCfg(
        name='stairs_left_right_foot_contact',
        primary=ContactMatch(mode='geom', pattern='left_foot[1-7]_collision', entity='robot'),
        secondary=ContactMatch(mode='body', pattern='right_ankle_roll_link', entity='robot'),
        secondary_policy='error', fields=('found', 'force', 'normal', 'tangent', 'dist', 'pos'),
        reduce='maxforce', num_slots=4, global_frame=True, history_length=cfg.decimation))
    cfg.scene.sensors = tuple(sensors)


class StairFailureState:
    """Per-foot stance state and per-episode events; all timers use seconds."""
    def __init__(self, count, device, options):
        validate(options)
        self.options = options
        self.loaded = torch.zeros(count, 2, dtype=torch.bool, device=device)
        self.safe_seen = self.loaded.clone()
        self.unload_used = self.loaded.clone()
        self.unloading = self.loaded.clone()
        self.edge = self.loaded.clone()
        self.multitread = self.loaded.clone()
        self.pair = torch.zeros(count, dtype=torch.bool, device=device)
        for name in ('safe_s', 'bad_s', 'multi_s', 'unload_s', 'load_ema', 'air_s'):
            setattr(self, name, torch.zeros(count, 2, device=device))
        self.pair_s = torch.zeros(count, device=device)
        self.pair_j = torch.zeros(count, device=device)

    def begin_step(self):
        self.edge.zero_(); self.multitread.zero_(); self.pair.zero_()

    def reset(self, selected):
        for value in vars(self).values():
            if torch.is_tensor(value):
                value[selected] = 0

    def observe(self, *, dt, load, weight, margin, clear_tread, multi_tread,
                unload_pose, pair_peak, pair_force_sum, armed, threshold):
        o = self.options
        enter = torch.maximum(weight * o['load_enter_fraction'], weight.new_tensor(o['minimum_load_n']))
        leave = torch.maximum(weight * o['load_exit_fraction'], weight.new_tensor(o['minimum_load_n'] / 2))
        self.loaded = torch.where(self.loaded, load >= leave[:, None], load >= enter[:, None])
        self.air_s = torch.where(self.loaded, 0., self.air_s + dt)
        airborne = self.air_s >= .02 - 1e-7
        self.safe_seen &= ~airborne
        self.unload_used &= ~airborne
        self.unloading &= ~airborne
        safe = self.loaded & clear_tread & (margin >= o['safe_margin_m'])
        self.safe_s = torch.where(safe, self.safe_s + dt, 0.)
        self.safe_seen |= self.safe_s >= o['safe_hold_s'] - 1e-7
        # A previously safe rear foot may roll off only after load transfer.
        falling = load < .85 * self.load_ema
        opposite_ready = (load.flip(-1) >= .20 * weight[:, None]) & self.safe_seen.flip(-1)
        start_unload = (self.safe_seen & ~self.unload_used & self.loaded & opposite_ready
                        & (load < .25 * weight[:, None]) & falling & unload_pose)
        self.unload_used |= start_unload
        self.unloading |= start_unload
        self.unload_s = torch.where(start_unload, 0., self.unload_s)
        self.unload_s += self.unloading * dt
        self.unloading &= (self.unload_s < o['unload_max_s']) & (load < .30 * weight[:, None])
        self.load_ema += (1 - math.exp(-dt / .02)) * (load - self.load_ema)
        eligible = self.loaded & armed[:, None] & ~self.unloading
        bad = eligible & clear_tread & (margin < threshold)
        self.bad_s = torch.where(bad, self.bad_s + dt, 0.)
        self.multi_s = torch.where(eligible & multi_tread, self.multi_s + dt, 0.)
        self.edge |= self.bad_s >= o['edge_hold_s'] - 1e-7
        self.multitread |= self.multi_s >= o['edge_hold_s'] - 1e-7
        light = (pair_peak >= o['pair_light_n']) & armed
        self.pair_s = torch.where(light, self.pair_s + dt, 0.)
        self.pair_j = torch.where(light, self.pair_j + pair_force_sum * dt, 0.)
        self.pair |= armed & ((pair_peak >= o['pair_peak_n']) |
                            ((self.pair_s >= o['pair_hold_s'] - 1e-7) &
                             (self.pair_j >= o['pair_impulse_ns'] - 1e-7)))


class StairFailureObserver:
    def __init__(self, env, quality, options):
        self.env, self.q, self.c, self.ids = env, quality, quality.c, quality.ids
        self.options = options
        self.state = StairFailureState(len(self.ids), env.device, options)
        self.last_step = -1
        self.age = torch.zeros(len(self.ids), device=env.device)
        self.edge_full = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.pair_full = self.edge_full.clone()
        self.target = self.edge_full.clone()
        self.invalid = torch.zeros(len(self.ids), dtype=torch.bool, device=env.device)
        self.metrics = {}
        self.snapshot = {}
        self.margin = torch.zeros(len(self.ids), 2, device=env.device)
        self.load = torch.zeros_like(self.margin)
        self.pair_peak = torch.zeros(len(self.ids), device=env.device)
        self.touchdowns = torch.zeros_like(self.margin)
        self.threshold = options['initial_margin_m']
        robot = env.scene['robot']
        self.body_ids = robot.indexing.body_ids
        self.gravity = float(torch.as_tensor(env.sim.mj_model.opt.gravity).norm())

    def begin_step(self):
        self.state.begin_step()
        self.invalid.zero_(); self.pair_peak.zero_(); self.touchdowns.zero_()
        update = int(getattr(self.env, 'policy_update_count', 0))
        o = self.options
        alpha = min(1., max(0., (update - o['start_update']) / max(1, o['adaptation_updates'])))
        self.threshold = o['initial_margin_m'] + alpha * (o['final_margin_m'] - o['initial_margin_m'])

    @torch.no_grad()
    def observe_substep(self):
        env, q, c, ids, o = self.env, self.q, self.c, self.ids, self.options
        tile = c.tile[ids]; origin = c.tile_origins[tile]
        edges, jumps, base = q.edges[tile], q.jumps[tile], q.base[tile]
        pos = c.robot_body_pos_w[ids][:, q.feet]
        quat = c.robot_body_quat_w[ids][:, q.feet]
        offsets = quat_apply(quat[:, :, None].expand(-1, -1, 4, -1).reshape(-1, 4),
                             q.sole_corners.expand(len(ids), 2, 4, 3).reshape(-1, 3)).reshape(len(ids), 2, 4, 3)
        corners = pos[:, :, None] + offsets
        x = corners[..., 0] - origin[:, None, None, 0]
        heights = torch.cat((base[:, None], base[:, None] + jumps.cumsum(-1)), -1).clamp_min(0)
        clear, multi, selected_height = [], [], []
        side_margin = 2. - (corners[..., 1] - origin[:, None, None, 1]).abs().amax(-1)
        for foot, side in enumerate(('left', 'right')):
            data = env.scene[f'foothold_{side}'].data
            found = data.found[ids] > 0
            finite = (torch.isfinite(data.force[ids]).all(-1) & torch.isfinite(data.pos[ids]).all(-1)
                      & torch.isfinite(data.normal[ids]).all(-1))
            self.invalid |= (found & ~finite).any(-1)
            valid = found & finite
            contact = torch.where(valid[..., None], data.pos[ids], 0.)
            force = torch.where(valid[..., None], data.force[ids], 0.)
            normal = torch.where(valid[..., None], data.normal[ids], 0.)
            local_x = contact[..., 0] - origin[:, None, 0]
            bins = (local_x[..., None] >= edges[:, None]).sum(-1)
            expected_z = heights.gather(1, bins) + origin[:, None, 2]
            horizontal = valid & (normal[..., 2].abs() > .7)
            horizontal &= (contact[..., 2] - expected_z).abs() <= o['height_tolerance_m']
            weights = torch.where(horizontal, force[..., 2].abs(), 0.)
            grouped = torch.zeros_like(heights).scatter_add_(1, bins, weights)
            total = grouped.sum(-1)
            dominant, index = grouped.max(-1)
            clear.append((total > 0) & (dominant >= o['dominance_fraction'] * total))
            meaningful = grouped > .15 * total[:, None]
            low = torch.where(meaningful, heights, torch.inf).amin(-1)
            high = torch.where(meaningful, heights, -torch.inf).amax(-1)
            multi.append((meaningful.sum(-1) >= 2) & (high - low > .015))
            lower = edges.gather(1, (index - 1).clamp_min(0)[:, None]).squeeze(-1)
            lower = torch.where(index > 0, lower, -1e4)
            upper = edges.gather(1, index.clamp_max(edges.shape[1] - 1)[:, None]).squeeze(-1)
            upper = torch.where(index < edges.shape[1], upper, 1e4)
            margin = torch.minimum(x[:, foot].amin(-1) - lower, upper - x[:, foot].amax(-1))
            self.margin[:, foot] = torch.minimum(margin, side_margin[:, foot])
            selected_height.append(heights.gather(1, index[:, None]).squeeze(-1) + origin[:, 2])
            net = env.scene[f'stairs_load_{side}'].data
            active = net.found[ids] > 0
            self.invalid |= (active & ~torch.isfinite(net.force[ids]).all(-1)).any(-1)
            vertical = torch.where(active, net.force[ids][..., 2], 0.).sum(-1).abs()
            self.load[:, foot] = torch.where(horizontal.any(-1), vertical, 0.)
        clear, multi = torch.stack(clear, -1), torch.stack(multi, -1)
        support_height = torch.stack(selected_height, -1)
        heel_height = corners[:, :, :2, 2].mean(-1) - support_height
        toe_height = corners[:, :, 2:, 2].mean(-1) - support_height
        linear, angular = c.robot_body_lin_vel_w[ids][:, q.feet], c.robot_body_ang_vel_w[ids][:, q.feet]
        heel_vz = (linear[:, :, None] + torch.linalg.cross(angular[:, :, None].expand_as(offsets), offsets, dim=-1))[:, :, :2, 2].mean(-1)
        unload_pose = (heel_height > .02) & (toe_height.abs() < .025) & (heel_vz > .03)
        pair = env.scene['stairs_left_right_foot_contact'].data
        found = pair.found[ids] > 0
        finite = torch.isfinite(pair.force[ids]).all(-1) & torch.isfinite(pair.dist[ids])
        self.invalid |= (found & ~finite).any(-1)
        magnitude = torch.where((found & finite)[..., None], pair.force[ids], 0.).norm(dim=-1)
        peak = magnitude.amax(-1)
        self.pair_peak = torch.maximum(self.pair_peak, peak)
        masses = env.sim.model.body_mass
        mass = masses[:, self.body_ids].sum(-1)
        weight = (mass.expand(env.num_envs) if len(mass) == 1 else mass)[ids] * self.gravity
        # A grace window is only available on the initial flat approach.
        nearest = (pos[..., 0, None] - origin[:, None, None, 0] - edges[:, None]).abs().amin((-1, -2))
        armed = (self.age >= o['startup_s']) | (nearest < o['startup_edge_distance_m'])
        before = self.state.loaded.clone()
        self.state.observe(dt=env.physics_dt, load=self.load, weight=weight, margin=self.margin,
                           clear_tread=clear, multi_tread=multi, unload_pose=unload_pose,
                           pair_peak=peak, pair_force_sum=magnitude.sum(-1), armed=armed,
                           threshold=self.threshold)
        self.touchdowns += self.state.loaded & ~before
        self.age += env.physics_dt

    @torch.no_grad()
    def finalize_step(self):
        step = int(self.env.common_step_counter)
        if self.last_step == step:
            return
        self.last_step = step
        if self.invalid.any():
            raise RuntimeError('Nonfinite active stair-failure contact; ordinary contact failures do not raise')
        kinds = self.c.tile_data['kind'][self.c.tile[self.ids]]
        mask = kinds != 3
        self.target.zero_(); self.target[self.ids] = mask
        edge = (self.state.edge | self.state.multitread).any(-1)
        self.edge_full.zero_(); self.pair_full.zero_()
        self.edge_full[self.ids] = edge & mask
        self.pair_full[self.ids] = self.state.pair & mask
        self.c.terminal_stair_edge = self.edge_full.clone()
        self.c.terminal_foot_collision = self.pair_full.clone()
        # These tensors survive auto reset until the wrapper has consumed them.
        self.snapshot = dict(edge=self.edge_full.clone(), pair=self.pair_full.clone(),
                             target=self.target.clone(), margin=self.margin.clone(),
                             load=self.load.clone(), pair_peak=self.pair_peak.clone(),
                             qpos=self.env.sim.data.qpos[self.ids].clone(),
                             root=self.c.robot.data.root_link_pos_w[self.ids].clone(),
                             tile=self.c.tile[self.ids].clone(), age=self.age.clone(),
                             raw_edge=edge.clone(), raw_pair=self.state.pair.clone(), threshold=self.threshold)
        denominator = mask.sum().clamp_min(1)
        self.metrics = {
            'StairFailure/edge_step_fraction': (edge & mask).sum() / denominator,
            'StairFailure/foot_collision_step_fraction': (self.state.pair & mask).sum() / denominator,
            'StairFailure/flat_edge_monitor_fraction': (edge & ~mask).float().mean(),
            'StairFailure/flat_pair_monitor_fraction': (self.state.pair & ~mask).float().mean(),
            'StairFailure/margin_threshold_m': self.threshold,
            'StairFailure/pair_peak_n': self.pair_peak.max(),
            'StairFailure/touchdowns': self.touchdowns.sum(),
            'StairFailure/unloading_fraction': self.state.unloading.float().mean(),
        }

    def reset(self, env_ids=None):
        all_ids = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.bool)
        if env_ids is None:
            all_ids[:] = True
        else:
            all_ids[env_ids] = True
        selected = all_ids[self.ids]
        self.state.reset(selected); self.age[selected] = 0


def edge_termination(env):
    env.stair_failures.finalize_step()
    return env.stair_failures.edge_full


def pair_termination(env):
    env.stair_failures.finalize_step()
    return env.stair_failures.pair_full


def failure_cost(env, base_func, base_params):
    env.stair_failures.finalize_step()
    physical = base_func(env, **base_params) > 0
    return (physical | env.stair_failures.edge_full | env.stair_failures.pair_full).float()


def guarded_success(env, base_func, base_params, fall_func, fall_params):
    env.stair_failures.finalize_step()
    success = base_func(env, **base_params)
    success &= ~(fall_func(env, **fall_params) | env.stair_failures.edge_full | env.stair_failures.pair_full)
    env.command_manager.get_term('motion').terminal_success = success.clone()
    return success
