"""Episode-constant, per-world colors independent of geometry and task labels."""
import copy
import torch
from mjlab.managers.event_manager import requires_model_fields

PALETTE = ((.93,.72,.035),(.45,.32,.025),(.95,.88,.42),(.08,.25,.8),
           (.75,.12,.08),(.5,.5,.5),(.12,.55,.22),(.65,.22,.65),(.85,.45,.12))

def validate_settings(settings):
    if settings.get('version') != 1:
        raise ValueError('Unknown appearance randomization version')
    if not 0 <= settings['nominal_fraction'] <= 1 or settings['ramp_updates'] <= 0:
        raise ValueError('Invalid appearance mixture/ramp')
    if settings['start_update'] < 0 or not 0 <= settings['independent_mat_fraction'] <= 1:
        raise ValueError('Invalid appearance start/mat fraction')


def validate_lighting_settings(settings):
    if settings.get('version')!=1 or not 0<=settings['nominal_fraction']<=1:
        raise ValueError('Invalid lighting version/mixture')
    for key in ('exposure_scale','gamma'):
        bounds=settings[key]
        if len(bounds)!=2 or not 0<bounds[0]<=bounds[1]<float('inf'):
            raise ValueError(f'Invalid lighting {key}')
    if not 0<=settings['color_balance']<1 or not 0<=settings['direction_xy_delta']<=1:
        raise ValueError('Invalid light color balance/direction range')


def sample_colors(n, device, settings, update, nominal):
    """Return floor/mat/pillar RGBA and the nominal-anchor mask."""
    validate_settings(settings)
    strength = min(1., max(0., (update-settings['start_update'])/settings['ramp_updates']))
    nominal_fraction = .7 + strength * (settings['nominal_fraction']-.7)
    anchor = torch.rand(n,device=device) < nominal_fraction
    palette=torch.tensor(PALETTE,device=device)
    ids=torch.randint(len(palette),(n,2),device=device)
    pillar=palette[ids[:,0]];mat=palette[ids[:,1]]
    # Continuous intensity/channel jitter covers shades between palette anchors.
    pillar=(pillar*torch.empty(n,1,device=device).uniform_(.65,1.15)+torch.empty(n,3,device=device).uniform_(-.04,.04)).clamp(.025,.98)
    mat=(mat*torch.empty(n,1,device=device).uniform_(.65,1.15)+torch.empty(n,3,device=device).uniform_(-.04,.04)).clamp(.025,.98)
    same=torch.rand(n,device=device)>=settings['independent_mat_fraction']
    mat=torch.where(same[:,None],pillar,mat)
    floor=torch.empty(n,1,device=device).uniform_(.22,.60).expand(n,3).clone()
    colors=torch.ones(n,3,4,device=device);colors[:,0,:3]=floor;colors[:,1,:3]=mat;colors[:,2,:3]=pillar
    colors=torch.where(anchor[:,None,None],nominal[None],colors)
    return colors,anchor


def _roles(env):
    """Terrain generator renames geoms; identify physical primitives, not colors."""
    cache=getattr(env,'_stone_appearance_roles',None)
    if cache is not None:return cache
    import mujoco
    model=env.sim.mj_model;gids=env.scene.terrain.indexing.geom_ids
    types=torch.as_tensor(model.geom_type,device=env.device)[gids]
    roles=torch.zeros_like(gids)
    roles[types==int(mujoco.mjtGeom.mjGEOM_BOX)]=1
    roles[types==int(mujoco.mjtGeom.mjGEOM_MESH)]=2
    # CylinderAtlas: only floor plane, mat boxes, and pillar meshes.
    allowed=(types==int(mujoco.mjtGeom.mjGEOM_PLANE))|(types==int(mujoco.mjtGeom.mjGEOM_BOX))|(types==int(mujoco.mjtGeom.mjGEOM_MESH))
    if not allowed.all() or not (roles==2).any():raise ValueError('Unexpected stone terrain geometry')
    mids=torch.as_tensor(model.geom_matid,device=env.device)[gids]
    material_roles=[]
    for mid in mids[mids>=0].unique().tolist():
        used=roles[mids==mid].unique()
        if len(used)!=1:raise ValueError('Terrain material shared across appearance roles')
        material_roles.append((mid,int(used[0])))
    nominal=torch.ones(3,4,device=env.device)
    default=env.sim.get_default_field('geom_rgba')[gids]
    for role in range(3):
        group=default[roles==role]
        if len(group):nominal[role]=group[0]
    cache=(gids,roles,material_roles,nominal);env._stone_appearance_roles=cache
    return cache


@requires_model_fields('geom_rgba','mat_rgba')
def randomize_appearance(env,env_ids,settings):
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    gids,roles,materials,nominal=_roles(env)
    colors,anchor=sample_colors(len(env_ids),env.device,settings,int(getattr(env,'policy_update_count',settings['start_update'])),nominal)
    env.sim.model.geom_rgba[env_ids[:,None],gids[None,:]]=colors[:,roles]
    for mid,role in materials:env.sim.model.mat_rgba[env_ids,mid]=colors[:,role]
    if not hasattr(env,'appearance_colors'):
        env.appearance_colors=nominal.expand(env.num_envs,3,4).clone()
        env.appearance_nominal=torch.ones(env.num_envs,device=env.device,dtype=torch.bool)
    env.appearance_colors[env_ids]=colors;env.appearance_nominal[env_ids]=anchor


def validate_appearance_migration(previous,current):
    old,new=copy.deepcopy(previous),copy.deepcopy(current)
    if old.get('appearance_randomization') or not new.get('appearance_randomization'):
        raise ValueError('Appearance migration must enable randomization once')
    validate_settings(new.pop('appearance_randomization'))
    # Yellow's last saved checkpoint predates the validated local-coordinate repair.
    if not old['stones'].get('local_coordinates',False) and new['stones'].get('local_coordinates') is True:
        new['stones'].pop('local_coordinates');old['stones'].pop('local_coordinates',None)
    old.pop('experiment_name',None);new.pop('experiment_name',None)
    if old!=new:raise ValueError('Appearance migration changed unrelated settings')


@requires_model_fields('light_dir')
def randomize_lighting(env,env_ids,settings):
    """Real supported light direction plus episode-constant camera photometrics.

    Warp 3.8 does not read MuJoCo diffuse/ambient coefficients. Exposure/gamma
    and color balance therefore act on the actual RGB passed to the encoder.
    """
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    n=len(env_ids);nl=env.sim.mj_model.nlight
    anchor=torch.rand(n,device=env.device)<settings['nominal_fraction']
    gain=torch.empty(n,device=env.device).uniform_(*settings['exposure_scale'])
    gamma=torch.empty(n,device=env.device).uniform_(*settings['gamma'])
    temperature=torch.empty(n,device=env.device).uniform_(-settings['color_balance'],settings['color_balance'])
    balance=torch.stack((1+temperature,torch.ones_like(temperature),1-temperature),-1)
    nominal=env.sim.get_default_field('light_dir')
    direction=nominal[None].expand(n,nl,3).clone()
    direction[:,:,:2]+=torch.empty(n,nl,2,device=env.device).uniform_(-settings['direction_xy_delta'],settings['direction_xy_delta'])
    direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    env.sim.model.light_dir[env_ids]=torch.where(anchor[:,None,None],nominal[None],direction)
    if not hasattr(env,'lighting_gain'):
        env.lighting_gain=torch.ones(env.num_envs,device=env.device)
        env.lighting_gamma=torch.ones(env.num_envs,device=env.device)
        env.lighting_balance=torch.ones(env.num_envs,3,device=env.device)
    env.lighting_gain[env_ids]=torch.where(anchor,1.,gain)
    env.lighting_gamma[env_ids]=torch.where(anchor,1.,gamma)
    env.lighting_balance[env_ids]=torch.where(anchor[:,None],1.,balance)


def photometric_rgb(rgb,env,ids):
    if not hasattr(env,'lighting_gain'):return rgb
    x=rgb.float()/255.
    x=x.pow(env.lighting_gamma[ids,None,None,None])
    x=x*env.lighting_gain[ids,None,None,None]*env.lighting_balance[ids,None,None,:]
    return (x.clamp(0.,1.)*255.).round().to(torch.uint8)
