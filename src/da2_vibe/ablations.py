"""Explicit contracts for controlled, raw-SONIC stair ablations."""
import copy

VARIANTS = ('full', 'no_geometry', 'no_history', 'blind', 'fixed_query')


def validate_ablation(settings):
    config = settings.get('ablation')
    if config is None:
        return
    if config.get('version') != 1 or config.get('variant') not in VARIANTS:
        raise ValueError('Unknown stairs ablation contract')
    comparison = settings.get('comparison', {})
    if (comparison.get('method') != 'da2_ours' or not comparison.get('stairs_only')
            or comparison.get('reference_mode') != 'all_flat_planner'
            or settings.get('initialization') != 'raw_sonic_fresh_visual'
            or settings.get('dagger') is not None or settings.get('native_vibe')):
        raise ValueError('Ablations require the raw-SONIC, stairs-only DA2 recipe')
    variant = config['variant']
    visual = settings['visual']
    expected_input = {'no_history': 'repeat_current', 'blind': 'constant_tokens'}.get(variant, 'normal')
    if (visual.get('ablation_input', 'normal') != expected_input
            or visual.get('query_mode', 'state') != ('constant' if variant == 'fixed_query' else 'state')
            or visual['frames'] != 2):
        raise ValueError('Visual implementation does not match the named ablation')
    expected_geometry = 0. if variant in ('no_geometry', 'blind') else .1
    if settings['geometry_aux']['coefficient'] != expected_geometry:
        raise ValueError('Geometry coefficient does not match the named ablation')
    if settings['expected_world_size'] not in (1, 4):
        raise ValueError('Stairs ablations support explicit one- or four-rank budgets')


def single_gpu_ablation(settings):
    validate_ablation(settings)
    return bool(settings.get('ablation')) and settings['expected_world_size'] == 1


def build_ablation(base, variant, *, world_size=1, checkpoint_index=2000):
    if variant not in VARIANTS or checkpoint_index < 0:
        raise ValueError('Invalid variant or checkpoint target')
    settings = copy.deepcopy(base)
    settings.update(ablation=dict(version=1, variant=variant),
        expected_world_size=world_size, iterations=checkpoint_index + 1)
    settings['visual'].update(ablation_input={
        'no_history': 'repeat_current', 'blind': 'constant_tokens'}.get(variant, 'normal'),
        query_mode='constant' if variant == 'fixed_query' else 'state')
    settings['geometry_aux']['coefficient'] = 0. if variant in ('no_geometry', 'blind') else .1
    settings['comparison']['budget_note'] = (
        f'{checkpoint_index + 1} updates; {world_size} ranks; '
        f'{settings["num_envs_per_rank"]} environments/rank; model_{checkpoint_index}.pt')
    validate_ablation(settings)
    return settings
