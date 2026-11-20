"""Read-only contact and geometric progress measurements for paired trials."""
import numpy as np


def completed_steps(trace, tile, tile_origin_x):
    """Count risers cleared by both feet while upright and supported, before failure."""
    feet_x = trace['foot_position_w'][..., 0].min(-1) - tile_origin_x
    root_x = trace['root_local'][:, 0]
    upright = (trace['projected_gravity_z'] < -.4) & (trace['root_ground_clearance_m'] > .25)
    supported = trace['supported'].any(-1)
    cleared_x = np.minimum(feet_x, root_x)
    counts = (cleared_x[:, None] >= np.asarray(tile['edges'])[None]).sum(-1)
    return int(np.where(upright & supported, counts, 0).max(initial=0)), len(tile['edges'])


def add_foot_pair_sensor(cfg):
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
        name='evaluation_foot_pair',
        primary=ContactMatch(mode='geom', pattern='left_foot[1-7]_collision', entity='robot'),
        secondary=ContactMatch(mode='body', pattern='right_ankle_roll_link', entity='robot'),
        secondary_policy='error', fields=('found', 'force', 'normal', 'tangent'), reduce='maxforce', num_slots=4,
        global_frame=True, history_length=cfg.decimation))


class FootPairObserver:
    """Measure real MuJoCo foot-to-foot forces at physics frequency; never terminate."""
    def __init__(self, env):
        import torch
        self.env = env
        self.peak = torch.zeros(env.num_envs, device=env.device)
        self.impulse = self.peak.clone()

    def begin_step(self):
        self.peak.zero_(); self.impulse.zero_()

    def observe_substep(self):
        import torch
        data = self.env.scene['evaluation_foot_pair'].data
        active = data.found > 0
        if (active & ~torch.isfinite(data.force).all(-1)).any():
            raise RuntimeError('Nonfinite active foot-to-foot contact')
        force = torch.where(active[..., None], data.force, 0.).norm(dim=-1)
        self.peak.copy_(torch.maximum(self.peak, force.amax(-1)))
        self.impulse += force.sum(-1) * self.env.physics_dt
