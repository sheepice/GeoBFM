def add_review_camera(cfg, tile):
    """World-fixed side camera encompassing only the robot and selected stairs."""
    import itertools
    import numpy as np
    from scipy.spatial.transform import Rotation
    from mjlab.sensor import CameraSensorCfg
    from ..velocity.step_quality_observer import tile_profile
    _, jumps, base = tile_profile(tile)
    height = max([base, *list(base+np.cumsum(jumps))])
    low, high = np.array([-.5,-.7,-.1]), np.array([tile['goal']+.6,.7,height+1.2])
    center=(low+high)/2
    z=np.array([-.15,-1.,.35]); z/=np.linalg.norm(z)
    x=np.cross([0.,0.,1.],z); x/=np.linalg.norm(x); y=np.cross(z,x)
    rotation=np.stack((x,y,z),axis=1)
    projected=(np.array(list(itertools.product(*zip(low,high))))-center)@rotation
    tangent=np.tan(np.deg2rad(43./2))
    distance=max(np.max(abs(projected[:,0])/(tangent*640/360)+projected[:,2]),
                 np.max(abs(projected[:,1])/tangent+projected[:,2]))*1.04
    size=cfg.scene.terrain.terrain_generator.size
    origin=np.array(tile['origin'])+[-size[0]/2,-size[1]/2,0.]
    if getattr(cfg.commands.get('motion'), 'local_coordinates', False):
        origin=np.zeros(3)
    cfg.scene.sensors=(*cfg.scene.sensors,CameraSensorCfg(name='review_camera',parent_body=None,
        pos=tuple(origin+center+distance*z),quat=tuple(Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]),
        width=640,height=360,fovy=43.,data_types=('rgb',),enabled_geom_groups=(0,2),use_shadows=False,use_textures=True))
