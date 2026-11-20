"""Opt-in GeoLoco light distributions in the actual Warp radiance calculation.

USD intensity units do not transfer to Warp. Divide each source intensity by
its GeoLoco nominal and color by 0.75, scaling the native ambient/direct terms.
Warp's hemisphere replaces the fixed HDR dome; it has no azimuthal texture.
Like GeoLoco, any nonempty reset changes the rank-global lights for all worlds.
"""
import copy
import hashlib
import importlib
from pathlib import Path

import torch
from mjlab.managers.event_manager import requires_model_fields

UPSTREAM_SHA256 = "f06e0df52854f2f3d01777b85dbcaf5f233a3a6c481b7c1bd8c41960ec2e70dc"


def validate(settings):
    if (settings.get('version') != 1 or settings.get('method') != 'geoloco_relative_radiance_v1'
            or settings.get('scope') != 'rank_global_reset'
            or settings.get('ambient_model') != 'warp_hemisphere_no_hdr'
            or settings.get('disable_robot_fill_light') is not True
            or settings.get('shadows') is not True or settings.get('retain_terrain_palette') is not True):
        raise ValueError('Unsupported render domain contract')
    for name in ('dome', 'sun'):
        lo, hi = settings[name + '_intensity']
        if not 0 < lo <= hi < float('inf') or not 0 < settings[name + '_nominal'] < float('inf'):
            raise ValueError('Invalid light intensity')
        if not 0 <= settings[name + '_color_variation'] <= 1:
            raise ValueError('Invalid light color variation')
    if settings['color_base'] != .75 or not 0 <= settings['sun_angle_deg'] <= 60:
        raise ValueError('Invalid light color/angle reference')


def validate_resume_options(previous, current, migrate=False):
    old, new = copy.deepcopy(previous), copy.deepcopy(current)
    if migrate:
        if old.get('render_domain') or not new.get('render_domain'):
            raise ValueError('Render migration must enable the domain exactly once')
        validate(new.pop('render_domain'))
    if old != new:
        raise ValueError('Resume changed unrelated terrain, camera, rewards or visual settings')


def sample_lights(settings, device, count=1):
    """Balanced RGB offsets follow GeoLoco; return radiance gains and rotations."""
    gains, intensities, colors = [], [], []
    for name in ('dome', 'sun'):
        intensity = torch.empty(count, 1, device=device).uniform_(*settings[name + '_intensity'])
        variation = settings[name + '_color_variation']
        offset = torch.empty(count, 3, device=device).uniform_(-variation, variation)
        color = (settings['color_base'] + offset - offset.mean(-1, keepdim=True)).clamp(0, 1)
        gains.append(intensity / settings[name + '_nominal'] * color / settings['color_base'])
        intensities.append(intensity)
        colors.append(color)
    angle = torch.empty(count, 3, device=device).uniform_(-settings['sun_angle_deg'], settings['sun_angle_deg'])
    return gains, intensities, colors, torch.deg2rad(angle)


def sun_direction(angles):
    """Three axis rotation factors applied to a downward directional source."""
    from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_apply
    q = quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])
    v = torch.zeros_like(angles); v[:, 2] = -1
    return quat_apply(q, v)


def install_renderer():
    """Process-local dispatch; ordinary simulations keep the upstream renderer."""
    import mujoco_warp as mjwarp
    import warp as wp
    from mjlab.sim import Simulation
    from mjlab.sensor.sensor_context import SensorContext
    if getattr(mjwarp.render, '_geoloco_dispatch', False):
        return
    source = importlib.import_module('mujoco_warp._src.render')
    if hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest() != UPSTREAM_SHA256:
        raise RuntimeError('Warp renderer version changed; re-audit the radiance adapter')
    from .geoloco_warp_render import render as domain_render
    from .groot_warp_render import render as groot_render
    original_render = mjwarp.render
    original_context = Simulation.set_sensor_context
    original_recreate = SensorContext.recreate

    def render(model, data, context):
        if hasattr(context, 'groot_state'):
            return groot_render(model, data, context)
        if hasattr(context, 'geoloco_ambient_gain'):
            return domain_render(model, data, context)
        return original_render(model, data, context)

    def set_context(sim, ctx):
        if getattr(sim.cfg, '_geoloco_render_domain', None):
            rc = ctx.render_context
            if not hasattr(rc, 'geoloco_ambient_gain'):
                # Allocate before graph capture and retain Torch ownership. Reset
                # modifies values in place so captured pointers remain valid.
                ambient = torch.ones(1, 3, device=sim.device)
                direct = torch.ones(sim.mj_model.nlight, 3, device=sim.device)
                rc.geoloco_tensors = (ambient, direct)
                rc.geoloco_ambient_gain = wp.from_torch(ambient, dtype=wp.vec3)
                rc.geoloco_direct_gain = wp.from_torch(direct, dtype=wp.vec3)
                if not rc.use_shadows or not rc.use_ambient_lighting:
                    raise RuntimeError('GeoLoco radiance requires shadows and ambient lighting')
            if getattr(sim.cfg, '_groot_visual_domain', None) and not hasattr(rc, 'groot_state'):
                from .groot_domain import attach_renderer
                attach_renderer(sim, rc)
        return original_context(sim, ctx)

    def recreate(ctx, *args, **kwargs):
        previous = ctx.render_context
        result = original_recreate(ctx, *args, **kwargs)
        if hasattr(previous, 'geoloco_tensors'):
            # EventManager expands per-world fields after initial construction.
            # Preserve radiance buffers when it replaces the render context,
            # before Simulation captures the replacement sense graph.
            for name in ('geoloco_tensors', 'geoloco_ambient_gain', 'geoloco_direct_gain'):
                setattr(ctx.render_context, name, getattr(previous, name))
            if hasattr(previous, 'groot_state'):
                from .groot_domain import RENDER_ATTRIBUTES
                for name in RENDER_ATTRIBUTES:
                    setattr(ctx.render_context, name, getattr(previous, name))
        return result

    render._geoloco_dispatch = True
    mjwarp.render = render
    Simulation.set_sensor_context = set_context
    SensorContext.recreate = recreate


def sun_id(env):
    if hasattr(env, '_geoloco_sun_id'):
        return env._geoloco_sun_id
    light_ids = env.scene.terrain.indexing.light_ids
    if len(light_ids) != 1:
        raise RuntimeError('Terrain must identify exactly one directional sun')
    light_id = int(light_ids[0])
    if int(env.sim.mj_model.light_type[light_id]) != 1:
        raise RuntimeError('Terrain sun must be directional')
    env._geoloco_sun_id = light_id
    return light_id


@requires_model_fields('light_castshadow', 'light_active')
def configure_light_rig(env, env_ids):
    light_id = sun_id(env)
    env.sim.model.light_castshadow[:, light_id] = True
    # G1 XML includes an unnamed trackcom fill light, absent on the real robot.
    # Keep this rig in nominal evaluation too, independent of random sampling.
    env.sim.model.light_active[:] = False
    env.sim.model.light_active[:, light_id] = True


@requires_model_fields('light_dir')
def randomize_lights(env, env_ids, settings):
    if env_ids is not None and len(env_ids) == 0:
        return
    rc = env.scene.sensor_context.render_context
    light_id = sun_id(env)
    gains, intensities, colors, angles = sample_lights(settings, env.device)
    rc.geoloco_tensors[0].copy_(gains[0])
    rc.geoloco_tensors[1][light_id].copy_(gains[1][0])
    env.sim.model.light_dir[:, light_id] = sun_direction(angles)
    env.render_domain_state = dict(intensities=intensities, colors=colors, angles=angles,
                                  sun_light_id=light_id)


def configure(cfg, settings):
    from dataclasses import replace
    from mjlab.managers import EventTermCfg
    validate(settings)
    install_renderer()
    cfg.sim._geoloco_render_domain = copy.deepcopy(settings)
    cfg.scene.sensors = tuple(replace(s, use_shadows=True) if s.name == 'head_cam' else s
                              for s in cfg.scene.sensors)
    cfg.events.pop('rand_light_dir', None)
    cfg.events['set_geoloco_light_rig'] = EventTermCfg(func=configure_light_rig, mode='startup')
    cfg.events['rand_geoloco_lighting'] = EventTermCfg(func=randomize_lights, mode='reset',
                                               params={'settings': copy.deepcopy(settings)})
    return cfg
