"""GR00T-inspired visual DR for Warp, explicitly opt-in and episode coherent.

HDRs drive SH diffuse illumination and the visible sky. Surface patterns and
roughness/metallic/specular shading replace Isaac's unsupported MDL materials.
No physics, geometry, camera intrinsics or policy architecture are changed.
"""
import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields


def validate(o):
    if (o.get('version') != 1 or o.get('method') != 'groot_hdr_material_rgb_v1'
            or o.get('sampling') != 'episode_coherent'
            or o.get('hdr_scope') != 'rank_global_reset'
            or o.get('hdr_min_steps') != 20):
        raise ValueError('Unsupported GR00T visual domain contract')
    if len(o.get('bank_sha256','')) != 64: raise ValueError('HDR bank must be pinned')
    for key in ('hdr_intensity','hdr_yaw','material_roughness','material_metallic','material_specular',
                'brightness','contrast','saturation','hue','noise_std','blur_sigma'):
        a,b=o[key]
        if not all(math.isfinite(x) for x in (a,b)) or a>b:
            raise ValueError(f'Invalid range: {key}')
    if (o['hdr_intensity'][0]<=0 or o['hdr_nominal']<=0
            or not 0<=o['hue'][0]+.5<=o['hue'][1]+.5<=1
            or any(not 0<=o[k][0]<=o[k][1]<=.8 for k in
                   ('material_roughness','material_metallic','material_specular'))
            or not 0<=o['nominal_fraction']<=1):
        raise ValueError('Invalid visual parameter bounds')
    for key in ('position_xyz_m','rotation_rpy_rad'):
        if len(o[key])!=3 or any(not 0<=x<=.2 for x in o[key]):
            raise ValueError('Invalid camera extrinsics')
    family=o.get('terrain_family','stairs')
    if family not in ('stairs','cylinder_stones'):
        raise ValueError('Unsupported visual terrain family')
    if o.get('level10_appearance') and family != 'cylinder_stones':
        raise ValueError('Level 10 appearance requires stone material groups')
    if family=='cylinder_stones':
        from ..da2_stones.appearance import validate_settings
        validate_settings(o['stone_appearance'])
    for k,p in o['probabilities'].items():
        if not 0<=p<=1: raise ValueError(f'Invalid probability: {k}')


def validate_resume_options(previous,current,migrate=False):
    old,new=copy.deepcopy(previous),copy.deepcopy(current)
    if not isinstance(old,dict) or not isinstance(new,dict):
        raise ValueError('Missing complete resume metadata')
    if migrate:
        if 'groot_visual' in old or 'groot_visual' not in new:
            raise ValueError('GR00T visual migration must add the domain exactly once')
        validate(new.pop('groot_visual'))
    if old!=new:
        raise ValueError('Visual migration changed unrelated camera, terrain, rewards or training settings')


def material_groups(model, family='stairs'):
    """Stable semantic render groups; robot remains group 2 for the shader."""
    import mujoco
    body=model.body('terrain').id
    terrain=model.geom_bodyid==body
    groups=np.full(model.ngeom,2,np.int32)
    if family=='cylinder_stones':
        # Terrain consists only of floor planes, mat boxes and pillar meshes.
        # Never use RGBA to distinguish same-yellow mats and pillars.
        plane=terrain & (model.geom_type==int(mujoco.mjtGeom.mjGEOM_PLANE))
        pillar=terrain & (model.geom_type==int(mujoco.mjtGeom.mjGEOM_MESH))
        mat=terrain & (model.geom_type==int(mujoco.mjtGeom.mjGEOM_BOX))
        if not (plane.any() and pillar.any() and mat.any()) or not np.array_equal(terrain,plane|pillar|mat):
            raise ValueError('Expected floor planes, pillar meshes and mat boxes')
        groups[plane],groups[pillar],groups[mat]=0,1,3
    elif family=='stairs':
        from orcs.tasks.perloco.terrain import FLOOR_RGBA
        groups[terrain]=1
        groups[terrain & np.isclose(model.geom_rgba,np.array(FLOOR_RGBA),atol=1e-5).all(-1)]=0
    else:
        raise ValueError('Unsupported visual terrain family')
    if not (groups==0).any() or not (groups==1).any():
        raise ValueError('Cannot identify both ground and obstacle render geometry')
    return groups


def attach_renderer(sim,rc):
    """Allocate before CUDA capture; reset only updates existing tensor storage."""
    import warp as wp
    o=sim.cfg._groot_visual_domain
    bank_path=Path(o['bank'])
    if hashlib.sha256(bank_path.read_bytes()).hexdigest()!=o['bank_sha256']:
        raise ValueError('HDR bank SHA256 mismatch')
    with np.load(bank_path) as bank:
        radiance=torch.tensor(bank['radiance'],device=sim.data.qpos.device)
        sh=torch.tensor(bank['irradiance_sh'],device=sim.data.qpos.device)
    if radiance.shape[1:]!=(64,128,3) or sh.shape!=(len(radiance),9,3):
        raise ValueError('Invalid HDR bank shape')
    device=sim.data.qpos.device;n=sim.data.qpos.shape[0];m=sim.mj_model
    groups=material_groups(m,o.get('terrain_family','stairs'))
    group_count=int(groups.max())+1
    state=dict(bank=radiance,sh_bank=sh,sh=sh[0].clone(),sky=radiance[0].clone(),
               controls=torch.tensor([[1.,0.,1.,0.]],device=device),
               colors=torch.ones(n,group_count,3,device=device),surface=torch.zeros(n,group_count,4,device=device),
               pattern=torch.zeros(n,group_count,4,device=device),groups=torch.tensor(groups,device=device),
               rgb=torch.zeros(n,8,device=device),active=False,last_step=None,
               reset_count=0,episode_count=0,settings=copy.deepcopy(o))
    if o.get('terrain_family')=='cylinder_stones':
        # sample_colors expects floor/mat/pillar, while renderer groups are
        # floor/pillar/robot/mat. Keep original per-role colors as anchors.
        nominal=np.stack([m.geom_rgba[np.flatnonzero(groups==g)[0],:3] for g in (0,3,1)])
        state['stone_nominal_rgba']=torch.cat((torch.as_tensor(nominal,device=device),torch.ones(3,1,device=device)),dim=-1)
        state['stone_appearance_nominal']=torch.ones(n,device=device,dtype=torch.bool)
        if o.get('level10_appearance'):
            state['level10_color_mode'] = torch.full((n,), -1, device=device, dtype=torch.long)
            state['level10_texture_group'] = torch.full_like(state['level10_color_mode'], -1)
    state['rgb'][:,:3]=1.;state['rgb'][:,6]=.5
    rc.groot_state=state
    rc.groot_sh=wp.from_torch(state['sh'],dtype=wp.vec3)
    rc.groot_sky=wp.from_torch(state['sky'],dtype=wp.vec3)
    rc.groot_controls=wp.from_torch(state['controls'],dtype=wp.vec4)
    rc.groot_colors=wp.from_torch(state['colors'],dtype=wp.vec3)
    rc.groot_surface=wp.from_torch(state['surface'],dtype=wp.vec4)
    rc.groot_pattern=wp.from_torch(state['pattern'],dtype=wp.vec4)
    rc.groot_groups=wp.from_torch(state['groups'],dtype=wp.int32)


RENDER_ATTRIBUTES=('groot_state','groot_sh','groot_sky','groot_controls','groot_colors',
                   'groot_surface','groot_pattern','groot_groups')


@requires_model_fields('light_dir')
def randomize_hdr(env,env_ids,settings,render_settings):
    if env_ids is not None and len(env_ids)==0:return
    s=env.scene.sensor_context.render_context.groot_state
    step=int(env.common_step_counter)
    if s['last_step'] is not None and step-s['last_step']<settings['hdr_min_steps']:return
    from .render_domain import randomize_lights
    randomize_lights(env,env_ids,render_settings)
    i=int(torch.randint(len(s['bank']),(1,),device=env.device))
    yaw=float(torch.empty(1,device=env.device).uniform_(*settings['hdr_yaw']))
    intensity=float(torch.empty(1,device=env.device).uniform_(*settings['hdr_intensity']))
    s['sh'].copy_(s['sh_bank'][i]);s['sky'].copy_(s['bank'][i])
    s['controls'].copy_(s['controls'].new_tensor([[math.cos(yaw),math.sin(yaw),intensity/settings['hdr_nominal'],1.]]))
    # New HDR exposure replaces the previous hemisphere exposure. Sun stays GeoLoco.
    env.scene.sensor_context.render_context.geoloco_tensors[0].fill_(1.)
    s.update(last_step=step,hdr_index=i,hdr_intensity=intensity,hdr_yaw=yaw,
             reset_count=s['reset_count']+1)


def sample_episode(n,device,o):
    def u(key):return torch.empty(n,device=device).uniform_(*o[key])
    def gated(key,identity):
        return torch.where(torch.rand(n,device=device)<o['probabilities'][key],u(key),identity)
    rgb=torch.stack((gated('brightness',1.),gated('contrast',1.),gated('saturation',1.),
                     gated('hue',0.),gated('noise_std',0.),
                     (torch.rand(n,device=device)<o['probabilities']['blur']).float(),
                     u('blur_sigma'),torch.zeros(n,device=device)),dim=-1)
    anchor=torch.rand(n,device=device)<o['nominal_fraction']
    rgb[anchor,:3]=1;rgb[anchor,3:6]=0
    return rgb


@requires_model_fields('cam_pos','cam_quat')
def randomize_episode(env,env_ids,settings):
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    if len(env_ids)==0:return
    from mjlab.utils.lab_api.math import quat_apply,quat_from_euler_xyz,quat_mul
    s=env.scene.sensor_context.render_context.groot_state;n=len(env_ids)
    cid=env.sim.mj_model.camera('head_cam').id
    def nominal(name):
        a=env.sim.get_default_field(name)
        return (a[cid][None].expand(n,-1) if a.ndim==2 else a[env_ids,cid]).contiguous()
    q0=nominal('cam_quat')
    pos=(torch.rand(n,3,device=env.device)*2-1)*q0.new_tensor(settings['position_xyz_m'])
    rpy=(torch.rand(n,3,device=env.device)*2-1)*q0.new_tensor(settings['rotation_rpy_rad'])
    # Local optical-frame perturbation about immutable nominal mounting; no drift.
    env.sim.model.cam_pos[env_ids,cid]=nominal('cam_pos')+quat_apply(q0,pos)
    env.sim.model.cam_quat[env_ids,cid]=quat_mul(q0,quat_from_euler_xyz(*rpy.unbind(-1)))
    s['rgb'][env_ids]=sample_episode(n,env.device,settings)
    # Independent obstacle/floor colors; stone roles have a separate palette.
    groups=s['colors'].shape[1]
    s['colors'][env_ids]=torch.empty(n,groups,3,device=env.device).uniform_(.12,.9)
    s['colors'][env_ids,2]=torch.empty(n,3,device=env.device).uniform_(.65,1.15)
    for j,key in enumerate(('material_roughness','material_metallic','material_specular')):
        s['surface'][env_ids,:,j]=torch.empty(n,groups,device=env.device).uniform_(*settings[key])
    s['surface'][env_ids,:,3]=torch.randint(0,4,(n,groups),device=env.device).float()
    s['pattern'][env_ids]=torch.rand(n,groups,4,device=env.device)
    s['pattern'][env_ids,:,0]=torch.empty(n,groups,device=env.device).uniform_(2.,15.)
    s['pattern'][env_ids,:,3]=torch.empty(n,groups,device=env.device).uniform_(.05,.35)
    if settings.get('terrain_family')=='cylinder_stones':
        from ..da2_stones.appearance import sample_colors
        palette=settings['stone_appearance']
        colors,anchor=sample_colors(n,env.device,palette,int(getattr(env,'policy_update_count',0)),s['stone_nominal_rgba'])
        for render_group,color_role in ((0,0),(1,2),(3,1)):
            s['colors'][env_ids,render_group]=colors[:,color_role,:3]
            # Nominal appearance samples retain plain original colors. Lighting
            # and material response remain randomized, as in the stairs domain.
            s['surface'][env_ids[anchor],render_group,3]=0.
        s['stone_appearance_nominal'][env_ids]=anchor
        if 'level10_color_mode' in s:
            s['level10_color_mode'][env_ids] = -1
            s['level10_texture_group'][env_ids] = -1
    s.update(active=True,episode_count=s['episode_count']+n)


@torch.no_grad()
def apply_level10_appearance(env, env_ids=None):
    """Select level 10 colors and texture once the current course is known."""
    state = env.scene.sensor_context.render_context.groot_state
    if not state['active'] or 'level10_color_mode' not in state:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    command = env.command_manager.get_term('motion')
    ids = env_ids[command.tile_data['level'][command.tile[env_ids]] == 10]
    count = len(ids)
    if count == 0:
        return

    from ..da2_stones.appearance import PALETTE

    device = env.device
    draw = torch.rand(count, device=device)
    mode = (draw >= .2).long() + (draw >= .5).long()
    nominal = state['stone_nominal_rgba'][:, :3]
    palette = nominal.new_tensor(PALETTE)
    colors = palette[torch.randint(len(palette), (count, 2), device=device)]
    gain = torch.empty(count, 2, 1, device=device).uniform_(.65, 1.15)
    jitter = torch.empty_like(colors).uniform_(-.04, .04)
    colors = (colors * gain + jitter).clamp_(.025, .98)
    pillar, mat = colors.unbind(1)
    anchor = mode == 0
    floor = torch.empty(count, 1, device=device).uniform_(.22, .60).expand(-1, 3)
    state['colors'][ids, 0] = torch.where(anchor[:, None], nominal[0], floor)
    state['colors'][ids, 1] = torch.where(anchor[:, None], nominal[2], pillar)
    mat = torch.where((mode == 1)[:, None], pillar, mat)
    state['colors'][ids, 3] = torch.where(anchor[:, None], nominal[1], mat)
    state['surface'][ids, 0, 3] = 0.
    state['surface'][ids, 1, 3] = 0.
    state['surface'][ids, 3, 3] = 0.
    state['stone_appearance_nominal'][ids] = anchor
    state['level10_color_mode'][ids] = mode

    textured = (mode != 0) & (torch.rand(count, device=device) < .15)
    texture_ids = ids[textured]
    texture_count = len(texture_ids)
    groups = 1 + 2 * torch.randint(2, (texture_count,), device=device)
    state['level10_texture_group'][ids] = -1
    state['level10_texture_group'][texture_ids] = groups
    state['surface'][texture_ids, groups, 3] = 4. + torch.randint(2, (texture_count,), device=device)
    params = torch.rand(texture_count, 4, device=device)
    params[:, 0].uniform_(12., 24.)
    params[:, 3].uniform_(.18, .25)
    state['pattern'][texture_ids, groups] = params


def augment_rgb(rgb,state,ids,*,noise=True):
    """Per-episode photometrics; new sensor noise each image; no chunk RNG gates."""
    if not state['active']:return rgb
    from torchvision.transforms import _functional_tensor as F
    x=rgb.permute(0,3,1,2).float()/255.
    p=state['rgb'][ids]
    x=(x*p[:,0,None,None,None]).clamp(0,1)
    gray=(x*x.new_tensor([.2989,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    x=(gray+p[:,2,None,None,None]*(x-gray)).clamp(0,1)
    hsv=F._rgb2hsv(x);hsv[:,0]=(hsv[:,0]+p[:,3,None,None])%1
    x=F._hsv2rgb(hsv)
    gray=(x*x.new_tensor([.2989,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    mean=gray.mean((-1,-2),keepdim=True)
    x=(mean+p[:,1,None,None,None]*(x-mean)).clamp(0,1)
    if noise:x=(x+torch.randn_like(x)*p[:,4,None,None,None]).clamp(0,1)
    # Correctly sample sigma, using a 5x5 kernel (GR00T functional call used 3x5).
    # Group convolution permits distinct episode-constant sigma for every image.
    b,c,h,w=x.shape
    axis=torch.arange(-2,3,device=x.device,dtype=x.dtype)
    kernel=torch.exp(-axis[None].square()/(2*p[:,6,None].square().clamp_min(.01)))
    kernel/=kernel.sum(-1,keepdim=True)
    weights=(kernel[:,:,None]*kernel[:,None,:])[:,None].repeat_interleave(c,0)
    blurred=torch.nn.functional.conv2d(torch.nn.functional.pad(x,(2,2,2,2),mode='reflect').reshape(1,b*c,h+4,w+4),
                                      weights,groups=b*c).reshape(b,c,h,w)
    x=torch.where(p[:,5,None,None,None]>.5,blurred,x)
    return (x.clamp(0,1)*255).round().byte().permute(0,2,3,1).contiguous()


def configure(cfg,settings):
    from mjlab.managers import EventTermCfg
    o=settings['groot_visual'];validate(o)
    if settings.get('appearance_randomization') or settings.get('lighting_randomization'):
        raise ValueError('GR00T visual domain replaces legacy appearance/lighting events')
    if 'stones' in settings and o.get('terrain_family')!='cylinder_stones':
        raise ValueError('Stone visual domain requires shape-based pillar/mat/floor groups')
    if not settings.get('render_domain'):raise ValueError('HDR extension requires radiance adapter')
    cfg.sim._groot_visual_domain=copy.deepcopy(o)
    for key in ('rand_cam_extrinsics','rand_terrain_color','rand_geoloco_lighting'):
        cfg.events.pop(key,None)
    cfg.events['rand_groot_hdr']=EventTermCfg(func=randomize_hdr,mode='reset',
        params={'settings':copy.deepcopy(o),'render_settings':copy.deepcopy(settings['render_domain'])})
    cfg.events['rand_groot_episode']=EventTermCfg(func=randomize_episode,mode='reset',
        params={'settings':copy.deepcopy(o)})
    return cfg
