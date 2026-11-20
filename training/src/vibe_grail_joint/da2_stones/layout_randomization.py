"""Many feasible layout descriptors sharing the same physical pillar meshes."""
import copy
import math
import numpy as np


def refresh(tile):
    solids=tile['solids']
    tile['height']=max(s['top'] for s in solids)
    tile['start']=solids[0]['x']-solids[0]['hx']
    tile['course_end']=solids[-1]['x']+solids[-1]['hx']
    tile['goal']=tile['course_end']+1.
    tile['tread']=float(np.mean(np.diff([s['x'] for s in solids])))
    distances=[math.hypot(b['x']-a['x'],b['y']-a['y']) for a,b in zip(solids,solids[1:])]
    tile['center_distances_m']=distances
    tile['surface_gaps_m']=[d-a['hx']-b['hx'] for d,a,b in zip(distances,solids,solids[1:])]


def yellow_layout(tile,rng,first_range,final,settings):
    """Rejection sampling jointly enforces the finite mat and spacing constraints."""
    stones=tile['solids'];count=len(stones);mat_start=tile['mats'][0]['x']-.5
    sign=1 if stones[0]['y']>0 else -1
    for _ in range(10000):
        first=rng.uniform(*first_range)
        if final:
            lateral=rng.uniform(*settings['lateral_offset_m'])
            # A scene has nearly equal distances; it is not 8 independent .6-.8 draws.
            mean=rng.uniform(*settings['center_distance_m'])
            ds=np.clip(mean+rng.uniform(-.035,.035,count-1),*settings['center_distance_m'])
        else:
            lateral=abs(stones[0]['y'])*rng.uniform(.92,1.08)
            nominal=np.array(tile['center_distances_m'])
            ds=nominal*rng.uniform(.85,1.03)*rng.uniform(.98,1.02,count-1)
        ys=sign*(-1.)**np.arange(count)*(lateral+rng.uniform(-.008,.008,count))
        if any(abs(y)+s['hx']>.495 for y,s in zip(ys,stones)):continue
        dy=np.diff(ys)
        if np.any(ds<=np.abs(dy)):continue
        xs=mat_start+np.r_[first,first+np.cumsum(np.sqrt(ds**2-dy**2))]
        if np.diff(xs).min()<.30:continue
        gaps=ds-np.array([a['hx']+b['hx'] for a,b in zip(stones,stones[1:])])
        if min(gaps)<=.025 or xs[-1]+stones[-1]['hx']>mat_start+4.99:continue
        if xs[0]-stones[0]['hx']<mat_start+.005:continue
        if final and (ds.max()-ds.min()>settings['max_distance_spread_m']+1e-8):continue
        for s,x,y in zip(stones,xs,ys):s['x'],s['y']=float(x),float(y)
        tile['stagger_m']=float(lateral);refresh(tile);return
    raise ValueError('Unable to fit requested nine-pillar layout within the five mats')


def build_layout_bank(physical_tiles,levels,base_variants,seed,settings):
    copies=settings['layouts_per_variant']
    if settings['version']!=1 or copies<4 or copies%4:raise ValueError('Use a layout bank divisible by four')
    rng=np.random.default_rng(seed+70819)
    tiles=[]
    for level in range(levels):
        # Mirror pairs share geometry, so difficulty does not leak first-foot choice.
        for k in range(copies):
            for pair in range(base_variants//2):
                source=level*base_variants+2*pair
                left=copy.deepcopy(physical_tiles[source]);left['physics_template']=source
                nominal=k<copies//4
                left['layout_group']='nominal' if nominal else 'target_offset' if k>=copies//2 else 'transition_offset'
                if not nominal:
                    if left.get('mats'):
                        first=settings['target_first_center_m'] if k>=copies//2 else settings['first_center_m']
                        yellow_layout(left,rng,first,level==levels-1,settings)
                    else:
                        stones=left['solids'];x=float(rng.uniform(1.1,2.2))+stones[0]['hx']
                        for j,s in enumerate(stones):
                            s['y']*=float(rng.uniform(.9,1.1))
                            if j:
                                a=stones[j-1];gap=float(np.clip(left['surface_gaps_m'][j-1]*rng.uniform(.8,1.2),.025,.14))
                                ds=a['hx']+s['hx']+gap;x=a['x']+math.sqrt(ds*ds-(s['y']-a['y'])**2)
                            s['x']=x
                        refresh(left)
                left['variant']=k*base_variants+2*pair
                right=copy.deepcopy(left);right['variant']+=1;right['physics_template']=source+1
                right['origin']=physical_tiles[source+1]['origin']
                if nominal:
                    right=copy.deepcopy(physical_tiles[source+1]);right.update(physics_template=source+1,variant=left['variant']+1,layout_group='nominal')
                else:
                    for stone in right['solids']:stone['y']=-stone['y']
                    if not left.get('mats'):
                        # Original random-course mirror templates have different
                        # radii/heights; preserve each compiled mesh's dimensions.
                        source_right=physical_tiles[source+1]
                        for j,stone in enumerate(right['solids']):
                            y=stone['y'];stone.update(copy.deepcopy(source_right['solids'][j]));stone['y']=y
                            if j==0:stone['x']=left['start']+stone['hx']
                            else:
                                a=right['solids'][j-1];ds=a['hx']+stone['hx']+left['surface_gaps_m'][j-1]
                                stone['x']=a['x']+math.sqrt(ds*ds-(stone['y']-a['y'])**2)
                        refresh(right)
                if left.get('mats'):
                    # Even the nominal XY anchors use the corrected mat height.
                    # Mirrored courses share the same sampled thickness.
                    thickness=float(rng.uniform(*settings['mat_thickness_m']))
                    for tile in (left,right):
                        old_thickness=tile['mats'][0]['top']
                        for mat in tile['mats']:
                            mat['bottom']=0.;mat['top']=thickness
                        for stone in tile['solids']:
                            stone['bottom']+=thickness-old_thickness
                            stone['top']+=thickness-old_thickness
                        tile['mat_thickness_m']=thickness
                        refresh(tile)
                tiles.extend((left,right))
    for i,t in enumerate(physical_tiles[levels*base_variants:]):
        tile=copy.deepcopy(t);tile['physics_template']=levels*base_variants+i;tile['layout_group']='flat';tiles.append(tile)
    return tuple(tiles)


def validate_domain_migration(previous,current):
    from .appearance import validate_settings
    old,new=copy.deepcopy(previous),copy.deepcopy(current)
    for key in ('appearance_randomization','lighting_randomization','layout_randomization'):
        if key in old or key not in new:raise ValueError('Domain migration enables a new documented distribution once')
    validate_settings(new.pop('appearance_randomization'))
    lighting=new.pop('lighting_randomization');layout=new.pop('layout_randomization')
    if lighting.get('version')!=1 or layout.get('version')!=1:raise ValueError('Unknown domain version')
    if layout['layouts_per_variant']%4 or layout['layouts_per_variant']<4:raise ValueError('Invalid layout bank')
    if not old['stones'].get('local_coordinates',False) and new['stones'].get('local_coordinates') is True:
        new['stones'].pop('local_coordinates');old['stones'].pop('local_coordinates',None)
    old.pop('experiment_name',None);new.pop('experiment_name',None)
    if old!=new:raise ValueError('Domain migration changed unrelated model, camera, reward or physics settings')
