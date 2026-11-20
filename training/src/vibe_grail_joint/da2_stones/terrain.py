"""Round pillars above a real floor; the approach starts at floor height."""
import math
import numpy as np
import torch
from ..stepping_stones.terrain import cylinder, PrimitiveHeights as PillarHeights


class PrimitiveHeights(PillarHeights):
    """Mats affect ground height but never count as a numbered pillar top."""
    def __init__(self, tiles, device):
        super().__init__(tiles, device)
        self.has_mats = any(t.get('mats') for t in tiles)
        if self.has_mats:
            # Mats form one contiguous, level rectangle in their own frame.
            frames = np.zeros((len(tiles), 7), dtype=np.float32)
            for i, tile in enumerate(tiles):
                mats = tile.get('mats', [])
                if mats:
                    yaw = mats[0].get('yaw_rad', 0.)
                    axis = np.array([math.cos(yaw), math.sin(yaw)])
                    centers = np.array([[m['x'], m['y']] for m in mats])
                    assert all(m['hy'] == .5 and m['top'] == mats[0]['top'] and m.get('yaw_rad', 0.) == yaw
                               for m in mats)
                    assert all(np.allclose(delta, (a['hx'] + b['hx']) * axis, atol=1e-6)
                               for delta, a, b in zip(np.diff(centers, axis=0), mats, mats[1:]))
                    start = centers[0] - mats[0]['hx'] * axis
                    end = centers[-1] + mats[-1]['hx'] * axis
                    frames[i] = [*((start + end) / 2), np.linalg.norm(end - start) / 2, .5, mats[0]['top'], *axis]
            self.mat_frames = torch.as_tensor(frames, device=device)

    def query(self, xy, tile, origins):
        height, surface = super().query(xy, tile, origins)
        if self.has_mats:
            expand = (slice(None),) + (None,) * (xy.ndim - 2)
            frame = self.mat_frames[tile][expand]
            q = xy - origins[tile, :2][expand] - frame[..., :2]
            x = q[..., 0] * frame[..., 5] + q[..., 1] * frame[..., 6]
            y = -q[..., 0] * frame[..., 6] + q[..., 1] * frame[..., 5]
            inside = (x.abs() <= frame[..., 2]) & (y.abs() <= frame[..., 3])
            ground = torch.where(inside, frame[..., 4], 0.)
            surface = torch.where(height > ground, surface, -1)
            height = torch.maximum(height, ground)
        return height, surface


def atlas_class():
    from dataclasses import dataclass
    import mujoco
    from ..terrain import JointAtlasCfg, STAIR_RGBA
    from mjlab.terrains.terrain_generator import TerrainGeometry

    @dataclass(kw_only=True)
    class CylinderAtlasCfg(JointAtlasCfg):
        random_tiles: tuple = ()

        def function(self, difficulty, spec, rng):
            result = super().function(difficulty, spec, rng)
            body = spec.body('terrain')
            if self.random_tiles and 'floor_rgba' in self.random_tiles[0]:
                for entry in result.geometries:
                    entry.geom.rgba = self.random_tiles[0]['floor_rgba']
                    entry.color = tuple(self.random_tiles[0]['floor_rgba'])
            for i, tile in enumerate(self.random_tiles):
                ox, oy, oz = tile['origin']
                color = tuple(tile.get('pillar_rgba', STAIR_RGBA))
                for j, mat in enumerate(tile.get('mats', [])):
                    mat_color = tuple(tile['mat_rgba'])
                    geom = body.add_geom(name=f'mat_{i}_{j}', type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=(mat['hx'], mat['hy'], mat['top'] / 2),
                        pos=(ox + mat['x'], oy + mat['y'], oz + mat['top'] / 2),
                        rgba=mat_color, friction=(.8, .005, .0001))
                    result.geometries.append(TerrainGeometry(geom=geom, color=mat_color))
                for j, stone in enumerate(tile['solids']):
                    bottom = stone.get('bottom', 0.)
                    count, r, height = stone['collision_sides'], stone['hx'], stone['top'] - bottom
                    angle = np.arange(count) * 2 * np.pi / count
                    ring = np.c_[r * np.cos(angle), r * np.sin(angle)]
                    # A fan from a slightly off-axis interior point avoids the
                    # old diameter seam where vertical mesh rays could miss all
                    # top triangles and incorrectly hit the cylinder bottom.
                    vertices = np.concatenate((np.c_[ring, np.full(count, -height / 2)],
                        np.c_[ring, np.full(count, height / 2)],
                        [[r * .0317, r * .0173, -height / 2], [r * .0317, r * .0173, height / 2]]))
                    faces = []
                    for k in range(count):
                        m = (k + 1) % count
                        faces.extend(((k, m, count + m), (k, count + m, count + k),
                                      (2 * count, m, k), (2 * count + 1, count + k, count + m)))
                    mesh = spec.add_mesh(name=f'round_stone_{i}_{j}', uservert=vertices.ravel().tolist(),
                                         userface=np.asarray(faces).ravel().tolist())
                    geom = body.add_geom(name=f'stone_{i}_{j}', type=mujoco.mjtGeom.mjGEOM_MESH,
                        meshname=mesh.name, pos=(ox + stone['x'], oy + stone['y'], oz + bottom + height / 2),
                        rgba=color, friction=(.8, .005, .0001))
                    result.geometries.append(TerrainGeometry(geom=geom, color=color))
            return result
    return CylinderAtlasCfg


def make_tiles(seed, levels=7, variants=4, real_course=None, minimum_stagger_m=0.):
    if real_course is not None:
        from .real_course import make_real_tiles
        return make_real_tiles(real_course, levels, variants)
    rng = np.random.default_rng(seed)
    tiles = []
    for level in range(levels):
        for variant in range(variants):
            count = 6 + 2 * level
            radius = .23 - .008 * level
            nominal_gap = .035 + .012 * level
            nominal_height = .08 + .022 * level
            irregular = max(0, level - 1) / max(1, levels - 2)
            stagger = max(minimum_stagger_m, 0. if level < 3 else .08 + .02 * (level - 3))
            # Both mirrored sequences are generated; each route alternates sides.
            sign = 1 if variant % 2 == 0 else -1
            start = 1.25 + (variant % 2) * .15
            solids = []
            gaps = []
            last_height = nominal_height
            for j in range(count):
                r = radius + rng.uniform(-.012, .012) * irregular
                y = sign * (-1) ** j * stagger
                gap = nominal_gap * (1 + rng.uniform(-.2, .2) * irregular)
                height = float(np.clip(last_height + rng.uniform(-.045, .045) * irregular,
                                       max(.065, nominal_height - .07), nominal_height + .07))
                if j == 0:
                    x = start + r
                else:
                    previous = solids[-1]
                    spacing = previous['hx'] + r + gap
                    x = previous['x'] + math.sqrt(spacing ** 2 - (y - previous['y']) ** 2)
                    gaps.append(gap)
                solid = cylinder(x, y, r, height)
                # Native cylinder contacts previously produced nonfinite EPA
                # states. A regular 64-sided convex cylinder preserves its round
                # footprint to <0.3mm and is used by physics AND analytic labels.
                solid['collision_sides'] = 64
                solids.append(solid)
                last_height = height
            end = solids[-1]['x'] + solids[-1]['hx']
            i = len(tiles)
            tiles.append(dict(kind='up_down', family='cylinder_stones', level=level,
                              course_level=level, variant=variant, n=count, start=start,
                              goal=end + 1., course_end=end, height=max(s['top'] for s in solids),
                              tread=float(np.mean(np.diff([s['x'] for s in solids]))),
                              plateau=0., base=0., edges=[], jumps=[], solids=solids,
                              surface_gaps_m=gaps, stagger_m=stagger,
                              origin=[-30. - (i % 8) * 16., -30. - (i // 8) * 8., 0.]))
    return tiles


def audit_tiles(tiles):
    for tile in tiles:
        if tile.get('family') == 'flat_training':
            assert not tile['solids'] and not tile.get('mats') and tile['n'] == 0
            continue
        stones = tile['solids']
        assert all(s['shape'] == 'cylinder' and s['collision_sides'] == 64 for s in stones)
        assert tile['base'] == 0 and not tile['edges']
        crossing = tile.get('route_kind', 0) > 0
        assert tile['goal'] < 12 and tile['start'] > (.5 if crossing else 1.)
        if crossing:
            target_x = [stones[i]['x'] for i in tile['route_targets']]
            assert target_x == sorted(set(target_x))
            assert tile['route_entry'] < target_x[0] <= target_x[-1] < tile['route_exit'] < tile['goal']
        for a, b in zip(stones, stones[1:]):
            gap = math.hypot(b['x'] - a['x'], b['y'] - a['y']) - a['hx'] - b['hx']
            real = tile.get('family') == 'real_yellow_nine_pillars'
            upper_dx=.8 if real and 'physics_template' in tile else .6
            lower_dx=.20 if 'physics_template' in tile else .25
            assert .02 < gap < (.51 if real else .15)
            assert abs(b['top'] - a['top']) <= (.151 if tile['level'] == 10 else .101 if real else .046)
            if not crossing:
                assert lower_dx < b['x'] - a['x'] < upper_dx
                assert not tile['stagger_m'] or a['y'] * b['y'] < 0
    return dict(tiles=len(tiles), levels=len(set(t['level'] for t in tiles if t.get('family') != 'flat_training')),
                all_cylinders=True, native_cylinder_collision=False, convex_cylinder_sides=64,
                approach_height_m=0., paired_stairs=0, hidden_bridges=0,
                mats_per_course=max(len(t.get('mats', [])) for t in tiles),
                final_pillars=max(t['n'] for t in tiles),
                mats_are_numbered_pillars=False)


def append_flat_tiles(tiles):
    """Two obstacle-free areas, separate from both old and yellow courses."""
    flat = [dict(kind='flat', family='flat_training', level=0, course_level=0, variant=i,
                 n=0, start=100., course_end=100., goal=100., height=0., tread=.3,
                 plateau=0., base=0., edges=[], jumps=[], solids=[], stagger_m=0.,
                 origin=[20., -100. - 10. * i, 0.]) for i in range(2)]
    return tuple(tiles) + tuple(flat)
