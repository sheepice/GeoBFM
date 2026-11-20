import copy
import torch
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from ..da2_locomotion.task import (
    QualityStairsEnv, REMOVED_REWARDS, original_mixture, task_fractions,
)
from ..da2_locomotion.task import options as base_options
from ..da2_locomotion.task import env_config as base_env_config, agent_config as base_agent_config


def edge_clearance_weight(foot_x, edges, jumps, full_distance=.6, zero_distance=.35):
    """Conservative distance from the entire foot span to either stair direction."""
    low, high = foot_x.amin(-1), foot_x.amax(-1)
    distance = torch.maximum(low[:, None] - edges, edges - high[:, None])
    relevant = torch.isfinite(edges) & (jumps.abs() > .015)
    distance = torch.where(relevant, distance, torch.inf).amin(-1)
    return ((distance - zero_distance) / (full_distance - zero_distance)).clamp(0, 1)


@torch.no_grad()
def flat_supervision(env, settings):
    c = env.command_manager.get_term('motion')
    feet = c.robot_body_pos_w[:, [3, 6]]
    origin = c.tile_origins[c.tile]
    local_x = feet[:, :, 0] - origin[:, None, 0]
    weight = edge_clearance_weight(local_x, c._profile_edges[c.tile], c._profile_jumps[c.tile],
                                   settings['full_edge_distance_m'], settings['zero_edge_distance_m'])
    # Require a flat physical support neighborhood, including toe/heel extents.
    offsets = feet.new_tensor([[-.17, -.08], [-.17, .08], [.17, -.08], [.17, .08]])
    points = (feet[:, :, None, :2] + offsets).flatten(1, 2)
    heights = c.ground_height(points)
    flat = heights.amax(-1) - heights.amin(-1) <= settings['height_tolerance_m']
    inside = ((feet[:, :, 1] - origin[:, None, 1]).abs() < 1.65).all(-1)
    weight *= (flat & inside & c.ppo_online).float()
    startup = env.episode_length_buf * env.step_dt < settings['startup_seconds']
    weight *= torch.where(startup, settings['startup_multiplier'], 1.)
    # This observation is deliberately excluded from BOTH actor and critic inputs.
    return weight[:, None]


def options(settings):
    result = base_options(settings)
    result['flat_repair'] = copy.deepcopy(settings['flat_repair'])
    return result


def env_config(bank_path, settings, num_envs):
    cfg = base_env_config(bank_path, settings, num_envs)
    # Keep physical stairs unchanged, but retain online episodes for a sustained
    # flat walk after ascending/descending instead of resetting just past them.
    tiles = []
    for original in cfg.commands['motion'].random_tiles:
        tile = copy.deepcopy(original)
        if tile['kind'] != 'flat':
            tile['goal'] += settings['flat_repair']['post_stair_flat_m']
        tiles.append(tile)
    cfg.commands['motion'].random_tiles = tuple(tiles)
    cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs'].random_tiles = tuple(tiles)
    cfg.observations['flat_supervision'] = ObservationGroupCfg(
        terms={'weight': ObservationTermCfg(func=flat_supervision,
                                           params={'settings': settings['flat_repair']})},
        concatenate_terms=True, enable_corruption=False)
    return cfg


def agent_config(bank_path, settings, num_envs):
    agent = base_agent_config(bank_path, settings, num_envs)
    agent['experiment_name'] = 'da2_flat_repair'
    agent['algorithm']['class_name'] = 'vibe_grail_joint.da2_flat_repair.algorithm.FlatRepairPPO'
    agent['algorithm']['flat_repair'] = settings['flat_repair']
    assert all('flat_supervision' not in groups for groups in agent['obs_groups'].values())
    return agent
