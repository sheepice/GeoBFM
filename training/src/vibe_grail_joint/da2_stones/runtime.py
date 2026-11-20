"""Physical contact quality and alternating top visits, without center targets."""
import torch
from mjlab.utils.lab_api.math import quat_apply
from ..velocity.step_quality_observer import StepQualityObserver
from .sequence import AlternatingSequence, loaded_top_bridge, lateral_foot_separation
from .landing import validate_settings as validate_landing, valid_top_support, swing_clearance_cost


class StoneGuidance:
    def __init__(self, env, settings):
        self.env, self.settings = env, settings
        self.c = env.command_manager.get_term('motion')
        self.clearance_cfg = validate_landing(settings)
        self.quality = StepQualityObserver(env)
        self.sequence = AlternatingSequence(env.num_envs, env.device)
        self.last_step = -1
        self.rewards = {name: torch.zeros(env.num_envs, device=env.device) for name in (
            'alternate', 'wrong_step', 'low_support', 'bridge', 'same_top',
            'riser_impact', 'shin_impact', 'slip', 'hard_landing', 'reach', 'crossover', 'wrong_side_swing', 'swing_clearance')}
        self.floor_failure = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.side_failure = self.floor_failure.clone()
        self.crossing_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.side_cfg = settings['stones'].get('side_aware')
        self.first_foot_by_tile = None
        if self.side_cfg:
            tiles = self.c.cfg.random_tiles
            if not hasattr(self.c, 'route_targets') and any(t['solids'] and abs(t['solids'][0]['y']) < 1e-6 for t in tiles):
                raise ValueError('Side-aware courses need an unambiguous left/right first pillar')
            self.first_foot_by_tile = torch.tensor([
                (0 if t['solids'][0]['y'] > 0 else 1) if t['solids'] else -1
                for t in tiles], device=env.device)
        self.metrics = {}
        self.exit_ground = torch.zeros_like(self.floor_failure)
        self.steps_since_progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        xy = torch.cartesian_prod(torch.linspace(-.055, .135, 9, device=env.device),
                                  torch.linspace(-.03, .03, 3, device=env.device))
        self.samples = torch.cat([xy, torch.full_like(xy[:, :1], -.035)], -1)
        self.terminal_progress = torch.zeros(env.num_envs, device=env.device)
        self.terminal_floor = self.floor_failure.clone()

    def before_step(self):
        self.quality.begin_step()

    @torch.no_grad()
    def measure(self):
        step = int(self.env.common_step_counter)
        if step == self.last_step:
            return
        self.last_step = step
        c, q = self.c, self.quality
        feet = c.robot_body_pos_w[:, [3, 6]]
        quat = c.robot_body_quat_w[:, [3, 6]]
        rotation = quat[:, :, None].expand(-1, -1, len(self.samples), -1)
        points = feet[:, :, None] + quat_apply(rotation.reshape(-1, 4),
            self.samples.expand(*rotation.shape[:-1], 3).reshape(-1, 3)).reshape(*rotation.shape[:-1], 3)
        h, sid = c.primitives.query(points[..., :2], c.tile, c.tile_origins)
        on_top = (sid >= 0) & ((points[..., 2] - h).abs() < .045)
        fraction = on_top.float().mean(-1)
        surfaces, supports, bridges = [], [], []
        routed = hasattr(c, 'route_targets')
        if routed:
            targets = c.route_targets[c.tile]
            required = c.route_count[c.tile]
            new_course = c.tile_data['level'][c.tile] == 10
            self.exit_ground.zero_()
            past_last_on_ground = torch.zeros_like(self.exit_ground)
        else:
            required = c.tile_data['n'][c.tile]
        floor = torch.zeros_like(self.floor_failure)
        for foot, side in enumerate(('left', 'right')):
            d = self.env.scene[f'foothold_{side}'].data
            height, contact_sid = c.primitives.query(d.pos[..., :2], c.tile, c.tile_origins)
            weights = torch.where((d.found > 0) & (d.normal[..., 2].abs() > .7),
                                  d.force[..., 2].abs(), 0.)
            loaded = weights > 15.
            top = loaded & (contact_sid >= 0) & ((d.pos[..., 2] - height).abs() < .015)
            # Select the strongest genuine top contact; do not require the
            # ankle or foot center to lie inside the pillar.
            strongest = torch.where(top, weights, 0.).argmax(-1)
            chosen = contact_sid.gather(-1, strongest[:, None]).squeeze(-1)
            chosen = torch.where(top.any(-1), chosen, -1)
            support = weights.sum(-1) > 20.
            bridge = loaded_top_bridge(contact_sid, top, weights)
            surfaces.append(chosen)
            supports.append(support)
            bridges.append(bridge & support)
            x = d.pos[..., 0] - c.tile_origins[c.tile, None, 0]
            inside_course = (x > c.course_start[c.tile, None] + .04) & (x < c.course_end[c.tile, None] - .04)
            if routed:
                on_ground = loaded & (contact_sid < 0) & (d.pos[..., 2] <= height + .015)
                after = x >= c.course_end[c.tile, None] - .04
                past_last_on_ground |= (on_ground & after).any(-1)
                beyond_mat = (c.route_kind[c.tile] == 0)[:, None] | (height <= 1e-5)
                self.exit_ground |= (on_ground & beyond_mat & (x >= c.route_exit[c.tile, None])
                                     & ((d.pos[..., 2] - height).abs() < .015)).any(-1)
            # A mat is real ground, not a foothold: landing on it between
            # pillars is a failure even if its height is above the old 3.5 cm cutoff.
            floor |= (loaded & inside_course & (contact_sid < 0) &
                      (d.pos[..., 2] <= height + .015)).any(-1)
        support = torch.stack(supports, -1)
        surfaces = torch.stack(surfaces, -1)
        physical_surfaces = surfaces
        if routed:
            exact_fraction = (on_top & (sid == surfaces[:, :, None])).float().mean(-1)
            fraction = torch.where(new_course[:, None], exact_fraction, fraction)
        bridge = torch.stack(bridges, -1)
        valid = valid_top_support(fraction, surfaces, self.settings['stones']['minimum_support_fraction'])
        active = c.pillar_task & (c.speed > .001) if hasattr(c, 'pillar_task') else c.online
        first_foot = self.first_foot_by_tile[c.tile] if self.side_cfg else None
        if routed:
            first_foot = c.route_first_foot[c.tile]
            # Convert physical pillar ID into its ordinal on this crossing.
            match = (surfaces[:, :, None] == targets[:, None, :]) & (surfaces[:, :, None] >= 0)
            ordinal = match.long().argmax(-1)
            off_route = valid & ~match.any(-1)
            ordinal = torch.where(match.any(-1), ordinal, -1)
            surfaces = torch.where(new_course[:, None], ordinal, surfaces)
            future_landing = valid & (surfaces > self.sequence.progress[:, None])
            self.side_failure |= active & new_course & (off_route | future_landing).any(-1)
        near_course = active & (c.robot.data.root_link_pos_w[:, 0] - c.tile_origins[c.tile, 0] > c.course_start[c.tile] - .65)
        near_course &= c.robot.data.root_link_pos_w[:, 0] - c.tile_origins[c.tile, 0] < c.course_end[c.tile] + .3
        crossing = torch.zeros_like(active)
        if self.side_cfg:
            separation = lateral_foot_separation(feet, c.robot.data.root_link_quat_w)
            margin = self.side_cfg['lateral_margin_m']
            self.rewards['crossover'] = ((margin - separation).clamp_min(0) / margin).square().clamp_max(4) * near_course
            crossing = (separation < 0) & near_course
            severe_crossing = (separation < -self.side_cfg['crossover_tolerance_m']) & near_course
            self.crossing_steps = torch.where(severe_crossing, self.crossing_steps + 1, 0)
        advance, wrong = self.sequence.step(support, surfaces, valid, bridge, first_foot, crossing)
        if routed:
            floor |= active & new_course & past_last_on_ground & (self.sequence.progress < required)
        if self.side_cfg:
            self.side_failure |= active & ((self.sequence.wrong_side > 0) |
                (self.crossing_steps >= self.side_cfg['crossover_failure_steps']))
        self.steps_since_progress = torch.where((advance > 0) | ~active, 0, self.steps_since_progress + 1)
        if hasattr(c, 'pillar_task'):
            floor &= c.pillar_task
        self.floor_failure.copy_(floor)
        self.terminal_progress.copy_(self.sequence.progress.float() / required.clamp_min(1))
        self.terminal_floor.copy_(floor)
        self.rewards['alternate'] = advance / self.env.step_dt
        self.rewards['wrong_step'] = wrong / self.env.step_dt
        on_pillar = support & (physical_surfaces >= 0)
        threshold = self.settings['stones']['minimum_support_fraction']
        self.rewards['low_support'] = (((threshold - fraction).clamp_min(0) / threshold).square() * on_pillar).mean(-1)
        self.rewards['bridge'] = bridge.float().mean(-1)
        self.rewards['same_top'] = (on_pillar.all(-1) & (physical_surfaces[:, 0] == physical_surfaces[:, 1])).float()
        data = q.step_data
        self.rewards['riser_impact'] = (data['foot_riser_impulse_ns'] / (self.env.step_dt * 100.)).clamp_max(3).mean(-1)
        self.rewards['shin_impact'] = (data['shin_impulse_ns'] / (self.env.step_dt * 100.)).clamp_max(3).mean(-1)
        self.rewards['slip'] = (((data['slip_distance_m'] / data['support_seconds'].clamp_min(1e-6) - .03).clamp_min(0) / .15).square().clamp_max(3) * support).mean(-1)
        self.rewards['hard_landing'] = (((-data['touchdown_vz_m_s'] / data['touchdowns'].clamp_min(1) - .8).clamp_min(0) / .8).square().clamp_max(3) * (data['touchdowns'] > 0)).mean(-1)
        # Shaping is zero everywhere in the supported disk: any well-supported
        # portion of the top is equally good. No distance-to-center objective.
        idx = self.sequence.progress.clamp_max(c.primitives.values.shape[1] - 1)
        if routed:
            slot = self.sequence.progress.clamp_max(targets.shape[1] - 1)
            physical_id = targets.gather(1, slot[:, None]).squeeze(1).clamp_min(0)
            idx = torch.where(new_course, physical_id, idx)
        target = c.primitives.values[c.tile, idx]
        target_xy = target[:, :2] + c.tile_origins[c.tile, :2]
        xy_error = ((feet[:, :, :2] - target_xy[:, None]).norm(dim=-1) - target[:, None, 2] + .025).clamp_min(0)
        can_swing = (~support) & ((self.sequence.last_foot[:, None] != torch.arange(2, device=feet.device)) | (self.sequence.last_foot[:, None] < 0))
        if self.side_cfg:
            expected = (first_foot + self.sequence.progress) % 2
            side_match = torch.arange(2, device=feet.device)[None] == expected[:, None]
            if routed:
                side_match |= (first_foot < 0)[:, None]
            can_swing &= side_match
            # Only a swing entering the next pillar's front edge is constrained;
            # ordinary flat approach steps can adjust the gait phase freely.
            entering = feet[:, :, 0] > target_xy[:, None, 0] - target[:, None, 2]
            raised = feet[:, :, 2] - c.tile_origins[c.tile, None, 2] > .085
            pending = self.sequence.progress < required
            self.rewards['wrong_side_swing'] = ((~support) & ~side_match & entering & raised).any(-1).float() * near_course * pending
        nearby = (c.robot.data.root_link_pos_w[:, 0] > target_xy[:, 0] - .65) & (self.sequence.progress < required)
        self.rewards['reach'] = (torch.exp(-xy_error.square() / .12 ** 2) * can_swing).amax(-1) * nearby
        clearance_active = self.clearance_cfg is not None
        if self.clearance_cfg:
            cost, shortfall, gate, _ = swing_clearance_cost(
                points, target, c.tile_origins[c.tile], can_swing,
                active & nearby & near_course, threshold, self.clearance_cfg)
            self.rewards['swing_clearance'] = cost
        from .failure_snapshot import assert_finite
        assert_finite(self)
        mask = c.pillar_task if hasattr(c, 'pillar_task') else c.online
        self.metrics = dict(progress_fraction=self.terminal_progress[mask].mean(),
            alternating_landings=advance.sum(), wrong_landings=wrong.sum(),
            first_foot_correct=self.sequence.first_correct.sum().float(),
            first_foot_wrong=self.sequence.first_wrong.sum().float(),
            wrong_side_landings=self.sequence.wrong_side.sum().float(),
            crossover_fraction=(crossing & active).sum() / active.sum().clamp_min(1),
            side_rule_failures=self.side_failure.sum().float(),
            floor_contacts=floor.sum().float(), bridge_fraction=bridge.float().mean(),
            same_top_fraction=self.rewards['same_top'].mean(),
            supported_fraction=(fraction * on_pillar).sum() / on_pillar.sum().clamp_min(1),
            foot_impulse_ns=data['foot_riser_impulse_ns'].sum(-1).mean(),
            shin_impulse_ns=data['shin_impulse_ns'].sum(-1).mean())
        if clearance_active:
            count = (gate[mask] > 0).sum().clamp_min(1)
            self.metrics.update(swing_clearance_cost=cost[mask].mean(),
                swing_clearance_active_fraction=(gate[mask] > 0).float().mean(),
                swing_clearance_shortfall_m=shortfall[mask].sum() / count)

    def reset(self, ids):
        self.quality.reset(ids)
        self.sequence.reset(ids)
        self.floor_failure[ids] = False
        self.side_failure[ids] = False
        self.crossing_steps[ids] = 0
        self.steps_since_progress[ids] = 0

    def reward(self, name):
        self.measure()
        return self.rewards[name]
