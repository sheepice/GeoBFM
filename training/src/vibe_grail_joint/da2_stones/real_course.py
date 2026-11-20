"""Measured nine-pillar target with an explicit easier-to-harder curriculum."""
import math
from ..stepping_stones.terrain import box, cylinder


def make_real_tiles(course, levels, variants):
    target = course['pillars_cm']
    stages = course['curriculum']
    if len(target) != 9 or levels != len(stages) or variants < 2 or variants % 2:
        raise ValueError('Real course requires nine pillars, all curriculum stages and mirrored variants')
    thickness = course['mat_thickness_m']
    mat_start = course['mat_start_x_m']
    size = course['mat_size_m']
    if course['mat_count'] != 5 or size != [1., 1.] or not 0 < thickness <= .05:
        raise ValueError('Expected five adjacent 1 m square thin mats')
    tiles = []
    for level, stage in enumerate(stages):
        for variant in range(variants):
            sign = 1 if variant % 2 == 0 else -1
            # The final stage is fixed, without random height/spacing perturbations.
            mats = [box(mat_start + j, mat_start + j + 1, -.5, .5, thickness)
                    for j in range(course['mat_count'])]
            stones = []
            for j, (diameter_cm, height_cm) in enumerate(target[:stage['count']]):
                diameter = stage['small_diameter_m'] if diameter_cm == 30 else stage['large_diameter_m']
                height = stage['end_height_m'] + stage['height_profile_scale'] * (height_cm / 100 - .15)
                x = mat_start + course['first_center_on_mat_m'] + j * stage['forward_spacing_m']
                y = sign * (-1) ** j * stage['lateral_offset_m']
                stone = cylinder(x, y, diameter / 2, thickness + height)
                stone.update(bottom=thickness, collision_sides=64, height_above_mat_m=height)
                if not (mat_start <= x - diameter / 2 and x + diameter / 2 <= mat_start + 5
                        and abs(y) + diameter / 2 <= .5):
                    raise ValueError('A pillar extends outside the five-mat footprint')
                stones.append(stone)
            distances = [math.hypot(b['x'] - a['x'], b['y'] - a['y']) for a, b in zip(stones, stones[1:])]
            gaps = [distance - a['hx'] - b['hx'] for distance, a, b in zip(distances, stones, stones[1:])]
            if min(gaps) <= .02:
                raise ValueError('Pillars must remain physically separated')
            end = stones[-1]['x'] + stones[-1]['hx']
            i = len(tiles)
            tiles.append(dict(kind='up_down', family='real_yellow_nine_pillars', level=level,
                course_level=level, variant=variant, n=len(stones),
                start=stones[0]['x'] - stones[0]['hx'], course_end=end, goal=end + 1.,
                height=max(s['top'] for s in stones), tread=stage['forward_spacing_m'],
                plateau=0., base=0., edges=[], jumps=[], solids=stones, mats=mats,
                center_distances_m=distances, surface_gaps_m=gaps,
                stagger_m=stage['lateral_offset_m'], target_pillars_cm=target,
                is_final_target=level == levels - 1, pillar_rgba=course['pillar_rgba'],
                mat_rgba=course['mat_rgba'], floor_rgba=course['floor_rgba'],
                origin=[-30. - (i % 8) * 16., -30. - (i // 8) * 8., 0.]))
    for tile in tiles[-variants:]:
        for stone, (diameter, height) in zip(tile['solids'], target, strict=True):
            if abs(stone['hx'] * 200 - diameter) > 1e-6 or abs(stone['height_above_mat_m'] * 100 - height) > 1e-6:
                raise ValueError('Final curriculum must exactly match the requested diameters/heights')
        if not all(.6 <= d <= .8 and .3 <= gap <= .5 for d, gap in
                   zip(tile['center_distances_m'], tile['surface_gaps_m'])):
            raise ValueError('Final target spacing falls outside the requested approximate range')
    return tiles
