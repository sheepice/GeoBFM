"""Express the active atlas course in its route frame independently in each world.

The atlas and course IDs stay unchanged. The terrain body and Warp's cached static
geom transforms move together; policy poses, contacts and camera rays stay near zero.
"""
import math

import torch


def exact_atlas_offset(generator):
    """The float64 corner translation used by TerrainGenerator itself.

    terrain_origins is a float32 runtime tensor; converting it back to double
    cannot recover the precision needed to bind compiled meshes in large atlases.
    CylinderAtlas returns a zero local spawn origin, so the grid corner is the
    exact model translation for the single generated atlas.
    """
    import numpy as np
    columns = len(generator.sub_terrains) if generator.curriculum else generator.num_cols
    return np.array([-generator.num_rows * generator.size[0] * .5,
                     -columns * generator.size[1] * .5, 0.], dtype=np.float64)


class LocalCourseCoordinates:
    def __init__(self, env, atlas_origins, tiles=None, physical_tiles=None):
        self.env = env
        self.atlas_origins = atlas_origins.clone()
        self.body_id = env.sim.mj_model.body('terrain').id
        env.sim.expand_model_fields(('body_pos',))
        self.base_position = env.sim.get_default_field('body_pos')[self.body_id].clone()
        # Warp caches world-welded geom transforms at make_data and skips them
        # in forward kinematics. Moving body_pos alone leaves the pillars behind.
        # Build translations in float64 before casting, so the active course's
        # centimeter geometry is never reconstructed by subtracting large f32s.
        model = env.sim.mj_model
        if model.body_parentid[self.body_id] != 0 or not (model.body_quat[self.body_id] == [1, 0, 0, 0]).all():
            raise ValueError('Local courses require an unrotated world-child terrain body')
        self.geom_ids = torch.as_tensor((model.geom_bodyid == self.body_id).nonzero()[0],
                                       device=atlas_origins.device)
        import mujoco
        reference = mujoco.MjData(model)
        mujoco.mj_kinematics(model, reference)
        global_positions = torch.from_numpy(reference.geom_xpos[self.geom_ids.cpu().numpy()].copy())
        # Atlas origins include the generator offset. The supplied f32 origins
        # define the translation contract; use it identically for robot/terrain.
        self.local_geom_positions = (global_positions[None] - atlas_origins.cpu().double()[:, None]).to(atlas_origins)
        self.mat_geom_ids = None
        self.mat_rotations = None
        if physical_tiles:
            # Reuse meshes, changing only their cached per-world positions.
            import numpy as np
            gids=self.geom_ids.cpu().numpy()
            centers=reference.geom_xpos[gids]
            offset=exact_atlas_offset(env.scene.terrain.cfg.terrain_generator)
            bindings={}
            for i,tile in enumerate(physical_tiles):
                indices=[]
                for stone in (*tile['solids'], *tile.get('mats', [])):
                    pos=np.array(tile['origin'])+offset+[stone['x'],stone['y'],(stone['top']+stone.get('bottom',0.))/2]
                    distances=np.linalg.norm(centers-pos,axis=-1);index=int(distances.argmin())
                    if distances[index]>2e-5:raise RuntimeError('Layout mesh binding does not match compiled terrain')
                    indices.append(index)
                bindings[i]=indices
            mat_indices=np.flatnonzero(model.geom_type[gids] == int(mujoco.mjtGeom.mjGEOM_BOX))
            if len(mat_indices):
                env.sim.expand_model_fields(('geom_size','geom_aabb','geom_rbound'))
                self.mat_geom_ids=self.geom_ids[torch.as_tensor(mat_indices,device=atlas_origins.device)]
                mat_columns={int(index):j for j,index in enumerate(mat_indices)}
                sizes=np.broadcast_to(model.geom_size[gids[mat_indices]],(len(tiles),len(mat_indices),3)).copy()
                if any('yaw_rad' in mat for tile in tiles for mat in tile.get('mats', [])):
                    base = reference.geom_xmat[gids[mat_indices]].reshape(len(mat_indices), 3, 3)
                    rotations = np.broadcast_to(base, (len(tiles), len(mat_indices), 3, 3)).copy()
            table=self.local_geom_positions.cpu().numpy()
            for i,tile in enumerate(tiles):
                source=physical_tiles[tile['physics_template']]
                for j,(stone,index) in enumerate(zip((*tile['solids'], *tile.get('mats', [])),bindings[tile['physics_template']],strict=True)):
                    original=(*source['solids'], *source.get('mats', []))[j]
                    if stone['hx']!=original['hx'] or stone['hy']!=original['hy']:
                        raise ValueError('Layout bank must preserve horizontal dimensions')
                    if j < len(tile['solids']):
                        if abs((stone['top']-stone.get('bottom',0.))-(original['top']-original.get('bottom',0.)))>1e-7:
                            raise ValueError('Layout bank must preserve pillar mesh height')
                    else:
                        column = mat_columns[index]
                        sizes[i,column,2]=(stone['top']-stone.get('bottom',0.))/2
                        if 'yaw_rad' in stone:
                            cos, sin = math.cos(stone['yaw_rad']), math.sin(stone['yaw_rad'])
                            rotations[i, column] = [[cos, -sin, 0.], [sin, cos, 0.], [0., 0., 1.]]
                    table[i,index]=[stone['x'],stone['y'],(stone['top']+stone.get('bottom',0.))/2]
            self.local_geom_positions=torch.as_tensor(table,device=atlas_origins.device)
            if self.mat_geom_ids is not None:
                self.mat_sizes=torch.as_tensor(sizes,device=atlas_origins.device,dtype=atlas_origins.dtype)
                if any('yaw_rad' in mat for tile in tiles for mat in tile.get('mats', [])):
                    self.mat_rotations = torch.as_tensor(rotations, device=atlas_origins.device, dtype=atlas_origins.dtype)

    def select(self, ids, tile):
        # The infinite floor translates with the pillars; Z is unchanged.
        self.env.sim.model.body_pos[ids, self.body_id] = self.base_position - self.atlas_origins[tile]
        self.env.sim.data.geom_xpos[ids[:, None], self.geom_ids[None, :]] = self.local_geom_positions[tile]
        if self.mat_geom_ids is not None:
            indices=(ids[:,None],self.mat_geom_ids[None,:])
            sizes=self.mat_sizes[tile]
            self.env.sim.model.geom_size[indices]=sizes
            # Broadphase bounds and narrowphase/render sizes must agree.
            self.env.sim.model.geom_aabb[indices[0],indices[1],0]=0.
            self.env.sim.model.geom_aabb[indices[0],indices[1],1]=sizes
            self.env.sim.model.geom_rbound[indices]=sizes.norm(dim=-1)
            if self.mat_rotations is not None:
                self.env.sim.data.geom_xmat[indices] = self.mat_rotations[tile]


def validate_coordinate_migration(previous, current):
    """Permit exactly the documented physics translation, never other overrides."""
    import copy
    old, new = copy.deepcopy(previous), copy.deepcopy(current)
    if old['stones'].get('local_coordinates', False) or new['stones'].get('local_coordinates') is not True:
        raise ValueError('Coordinate migration must enable local coordinates once')
    new['stones'].pop('local_coordinates')
    old['stones'].pop('local_coordinates', None)
    old.pop('experiment_name', None)
    new.pop('experiment_name', None)
    if old != new:
        raise ValueError('Coordinate migration also changed unrelated training settings')


def audit_local_coordinates(command):
    if not getattr(command.cfg, 'local_coordinates', False):
        return dict(enabled=False)
    frame = command.local_frame
    actual = command._env.sim.model.body_pos[:, frame.body_id]
    expected = frame.base_position - frame.atlas_origins[command.tile]
    if not torch.equal(actual, expected) or command.tile_origins.abs().max() != 0:
        raise RuntimeError('Active physical terrain and analytic course frames differ')
    geom = command._env.sim.data.geom_xpos[:, frame.geom_ids]
    if not torch.equal(geom, frame.local_geom_positions[command.tile]):
        raise RuntimeError('Static terrain geometry did not follow the course translation')
    if frame.mat_geom_ids is not None:
        sizes=frame.mat_sizes[command.tile]
        model=command._env.sim.model
        if (not torch.equal(model.geom_size[:,frame.mat_geom_ids],sizes)
                or not torch.equal(model.geom_aabb[:,frame.mat_geom_ids,1],sizes)
                or not torch.equal(model.geom_rbound[:,frame.mat_geom_ids],sizes.norm(dim=-1))):
            raise RuntimeError('Mat collision/render sizes and broadphase bounds differ')
        if frame.mat_rotations is not None and not torch.equal(
                command._env.sim.data.geom_xmat[:, frame.mat_geom_ids], frame.mat_rotations[command.tile]):
            raise RuntimeError('Material and collision frames have different mat orientations')
    return dict(enabled=True, terrain_body_id=frame.body_id,
                independent_world_translation=True, course_origins_zero=True,
                root_xy_max_m=float(command.robot.data.root_link_pos_w[:, :2].abs().max()))
