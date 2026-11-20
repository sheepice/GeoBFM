"""Multi-GPU pillar training, optionally mixed with flat stand/start-stop tasks."""
from da2_vibe.paths import load_checkpoint
from da2_vibe.paths import load_settings, profile
import argparse
import copy
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path
from ..gpu_guard import verify_guarded_worker


def task_fractions(settings):
    if settings.get('stones_stairs_style'):
        return dict(pillar_online_ppo=.6, flat_planner_dagger=.2, flat_replay_dagger=.2, paired_stairs=0.)
    if settings.get('flat_training'):
        if settings.get('flat_dagger'):
            fractions = dict(pillar_online_ppo=.6, flat_standing_dagger=.15,
                             flat_start_stop_dagger=.15, paired_stairs=0., flat_replay=0.)
            fractions['flat_walking_dagger' if 'walk' in settings['flat_dagger']['roles'] else 'flat_walking_ppo'] = .10
            return fractions
        if settings.get('flat_standing_dagger'):
            return dict(pillar_online_ppo=.6, flat_standing_dagger=.15,
                        flat_walking_ppo=.10, flat_start_stop_ppo=.15,
                        paired_stairs=0., flat_replay=0.)
        return dict(pillar_online_ppo=.6, flat_standing_ppo=.15, flat_walking_ppo=.10,
                    flat_start_stop_ppo=.15, paired_stairs=0., flat_replay=0., dagger=0.)
    return dict(pillar_online_ppo=1., paired_stairs=0., flat_replay=0., dagger=0.)


def wrap_training_env(collector, settings, rank):
    """Route all supported DAgger recipes through PPO-only rollout slicing."""
    from .stairs_style import enabled, StairsStyleDaggerEnv
    if enabled(settings):
        return StairsStyleDaggerEnv(collector, settings, rank)
    from .flat_dagger import validate_settings, FlatDaggerEnv
    if validate_settings(settings):
        return FlatDaggerEnv(collector, settings, rank)
    return collector


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--log-dir', required=True)
    p.add_argument('--num-envs', type=int)
    p.add_argument('--iterations', type=int)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-rollout-steps', type=int, default=24)
    p.add_argument('--resume')
    p.add_argument('--init-actor', help='Warm start actor when changing flat PPO/DAgger roles')
    p.add_argument('--migrate-local-coordinates', action='store_true')
    p.add_argument('--migrate-appearance', action='store_true')
    p.add_argument('--migrate-domain', action='store_true')
    args = p.parse_args()
    if args.init_actor and (args.resume or args.migrate_domain or args.migrate_appearance or args.migrate_local_coordinates):
        raise ValueError('Actor initialization cannot combine with resume/migration')
    if args.migrate_domain and (not args.resume or args.migrate_appearance or args.migrate_local_coordinates):
        raise ValueError('Domain migration requires resume and cannot combine migration modes')
    if args.migrate_appearance and (not args.resume or args.migrate_local_coordinates):
        raise ValueError('Appearance migration requires resume and owns the optional local-coordinate migration')
    if args.migrate_local_coordinates and not args.resume:
        raise ValueError('Coordinate migration requires a complete source checkpoint')
    if not 24 <= args.smoke_rollout_steps <= 128 or (not args.smoke and args.smoke_rollout_steps != 24):
        raise ValueError('Longer rollouts are only permitted in bounded smoke validation (24..128 steps)')
    verify_guarded_worker()
    import mjlab
    import torch
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from ..catalog import write_json
    from ..common.checkpoint import frozen_digest
    from ..common.audit import distributed_parameter_audit, shared_initial_digest
    from ..common.cuda_graph import build_sensor_env
    from ..da2_locomotion.collection import GeometryCollector, audit_terrain_labels
    from .task import env_config, agent_config, StoneEnv
    from .initialization import load_stage1, audit_raw_sonic
    from .terrain import audit_tiles

    settings = load_settings(args.config)
    from .comparison import enabled as comparison_enabled, audit_actor
    comparison = comparison_enabled(settings)
    matched_vibe = comparison and settings['stones_comparison']['method'] == 'vibe_matched'
    native_vibe = comparison and settings['stones_comparison']['method'] == 'vibe_native_ppo'
    raw_sonic = settings.get('initialization') == 'raw_sonic_fresh_visual'
    level10 = settings.get('level10') is not None
    if level10 and not (args.init_actor or args.resume):
        raise ValueError('Level 10 requires the trained nine-pillar actor via --init-actor or its own --resume')
    if raw_sonic and not comparison:
        raise ValueError('Raw SONIC requires an explicit pillar comparison protocol')
    from .flat_dagger import validate_settings, validate_actor_initialization, FlatDaggerEnv, dagger_roles, role_mask
    standing = validate_settings(settings)
    from .stairs_style import enabled, validate_settings as validate_stairs_style, StairsStyleDaggerEnv
    stairs_style = enabled(settings)
    validate_stairs_style(settings)
    if stairs_style and ((args.init_actor and not level10) or args.migrate_domain or args.migrate_appearance or args.migrate_local_coordinates):
        raise ValueError('Changed command/replay objective requires a fresh stage1 run; do not migrate legacy optimizer/replays')
    if args.init_actor and not (standing or level10):
        raise ValueError('--init-actor requires a flat DAgger configuration')
    dagger_replay_key = ('stairs_style_dagger' if stairs_style else
                         'flat_dagger' if settings.get('flat_dagger') else 'standing_dagger')
    rank, local, world = [int(os.environ.get(k, d)) for k, d in [('RANK', '0'), ('LOCAL_RANK', '0'), ('WORLD_SIZE', '1')]]
    if world < 2 or (not args.smoke and world != settings['expected_world_size']):
        raise ValueError('Training requires the configured world size and at least two ranks')
    if args.smoke and (args.iterations or settings['iterations']) > 10:
        raise ValueError('Smoke must be bounded to <=10 actual updates')
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    torch.manual_seed(settings['seed'] + rank)
    device = f'cuda:{local}'
    n = args.num_envs or settings['num_envs_per_rank']
    iterations = args.iterations or settings['iterations']
    out = Path(args.log_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / f'rank_{rank}_status.json'
    if status_path.exists() and not args.resume:
        raise ValueError(f'Run already exists: {out}')
    record = dict(status='initializing', rank=rank, world_size=world, num_envs=n, smoke=args.smoke,
                  requested_iterations=iterations, settings=settings, completed_updates=0)
    runtime = profile().get('runtime', {})
    disable_graph_capture = bool(runtime.get('disable_cuda_graph_capture', False))
    disable_mempool = bool(runtime.get('disable_warp_mempool', False))
    record['runtime'] = dict(disable_cuda_graph_capture=disable_graph_capture)
    write_json(status_path, record)
    start = time.monotonic()
    env = None

    class LoggingWrapper(RslRlVecEnvWrapper):
        def __init__(self, raw):
            super().__init__(raw)
            self.metric_sums = {}
            self.metric_steps = 0

        def step(self, actions):
            result = super().step(actions)
            c = self.unwrapped.command_manager.get_term('motion')
            metrics = dict(self.unwrapped.guidance.metrics)
            if hasattr(self.unwrapped, 'appearance_nominal'):
                metrics['appearance_nominal_fraction'] = self.unwrapped.appearance_nominal.float().mean()
            if hasattr(self.unwrapped, 'lighting_gain'):
                metrics['lighting_exposure_gain'] = self.unwrapped.lighting_gain.mean()
            if settings.get('groot_visual'):
                visual_state=self.unwrapped.scene.sensor_context.render_context.groot_state
                metrics.update(visual_hdr_resamples=float(visual_state['reset_count']),
                    visual_episode_samples=float(visual_state['episode_count']),
                    visual_hdr_index=float(visual_state.get('hdr_index',-1)),
                    visual_hdr_intensity=float(visual_state.get('hdr_intensity',0)),
                    visual_nominal_appearance_fraction=visual_state['stone_appearance_nominal'].float().mean(),
                    visual_brightness_fraction=(visual_state['rgb'][:,0]!=1).float().mean(),
                    visual_blur_fraction=visual_state['rgb'][:,5].mean(),
                    visual_noise_fraction=(visual_state['rgb'][:,4]>0).float().mean())
            metrics['speed_error_m_s'] = (c.robot.data.root_link_lin_vel_w[:, 0] - c.speed).abs().mean()
            if stairs_style:
                idle = c.pillar_task & (c.speed < .001)
                count = idle.sum().clamp_min(1)
                speed = c.robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
                drift = (c.robot.data.root_link_pos_w[:, :2] - c.idle_anchor).norm(dim=-1)
                metrics.update(obstacle_idle_fraction=idle.float().mean(),
                    obstacle_idle_speed_m_s=(speed * idle).sum() / count,
                    obstacle_idle_drift_m=(drift * idle).sum() / count,
                    obstacle_idle_upright=((-c.robot.data.projected_gravity_b[:, 2]) * idle).sum() / count)
            elif hasattr(c, 'flat_task'):
                idle = c.flat_task & (c.speed < .001)
                count = idle.sum().clamp_min(1)
                v = c.robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
                metrics.update(flat_fraction=c.flat_task.float().mean(),
                    idle_fraction=idle.float().mean(),
                    idle_speed_m_s=(v * idle).sum() / count,
                    idle_upright=((-c.robot.data.projected_gravity_b[:, 2]) * idle).sum() / count,
                    flat_speed_error_m_s=((c.robot.data.root_link_lin_vel_w[:, 0] - c.speed).abs() * c.flat_task).sum() / c.flat_task.sum().clamp_min(1),
                    start_transitions=c.start_transitions.float(), stop_transitions=c.stop_transitions.float())
            for k, value in metrics.items():
                self.metric_sums[k] = self.metric_sums.get(k, 0.) + torch.as_tensor(value,device=self.device).detach()
            self.metric_steps += 1
            result[-1].setdefault('log', {}).update({'Stones/' + k: v for k, v in metrics.items()})
            return result

    class Runner(MjlabOnPolicyRunner):
        def load(self, path, *args, **kwargs):
            import io
            from da2_vibe.paths import load_checkpoint
            state = load_checkpoint(path, map_location='cpu', weights_only=False)
            buffer = io.BytesIO()
            torch.save(state, buffer)
            buffer.seek(0)
            return super().load(buffer, *args, **kwargs)

        def _configure_multi_gpu(self):
            from ..velocity.recovery_support import configure_distributed
            configure_distributed(self, settings['collective_timeout_seconds'])

        def save(self, path, infos=None):
            if frozen_digest(self.alg.get_policy()) != frozen_before:
                raise RuntimeError('Frozen SONIC changed')
            metadata = dict(infos or {}, da2_stones=dict(version=1, settings=settings,
                stage1=record['initialization'], world_size=world, terrain_hashes=terrain_hashes,
                fractions=task_fractions(settings),
                flat_teacher=('Student-driven DAgger for ' + ('standing/walking/start-stop'
                              if 2 in dagger_roles(standing) else 'standing/start-stop; walking retains PPO and teacher MSE'
                              if 3 in dagger_roles(standing) else 'standing; walking/start-stop retain PPO and teacher MSE')
                              if standing else 'Frozen SONIC labels current student flat states; auxiliary to PPO')
                             if settings.get('flat_training') else
                             'Original stairs online-planner and flat-source replay DAgger' if stairs_style else None),
                da2_locomotion=dict(version=1, settings=settings['visual'], camera=settings['camera']))
            temp = str(path) + '.tmp'
            super().save(temp, metadata)
            os.replace(temp, path)

    try:
        bank = str(Path(settings['bank_root']) / f'rank_{rank}') if stairs_style else settings['schema_bank']
        cfg = env_config(bank, settings, n, rank)
        raw = build_sensor_env(StoneEnv, cfg, device=device, settings=settings,
                               disable_graph_capture=disable_graph_capture,
                               disable_warp_mempool=disable_mempool)
        if disable_mempool:
            import warp as wp
            record['runtime']['warp_mempool_enabled'] = wp.is_mempool_enabled(device)
        record['runtime']['cuda_graphs_active'] = bool(raw.sim.use_cuda_graph)
        raw.failure_snapshot_dir = out
        from ..da2_locomotion.camera import camera_contract, assert_sim_camera
        record['actual_camera'] = assert_sim_camera(raw, camera_contract(next(s for s in cfg.scene.sensors if s.name == 'head_cam')))
        wrapped = LoggingWrapper(raw)
        collector = GeometryCollector(wrapped, settings['geometry_aux'], rank,
                                      enabled=settings['geometry_aux']['coefficient'] > 0)
        env = wrap_training_env(collector, settings, rank)
        dagger_env = env if env is not collector else None
        agent = agent_config(bank, settings, n)
        if args.smoke:
            agent['save_interval'] = 1
            agent['num_steps_per_env'] = args.smoke_rollout_steps
            if not native_vibe:
                agent['algorithm']['num_mini_batches'] = max(1, math.ceil(env.num_envs * args.smoke_rollout_steps / settings['minibatch_env_steps']))
        action = raw.action_manager.get_term('joint_pos')
        scales = torch.as_tensor(action.scale, device=device)
        if scales.ndim == 0:
            scales = scales.expand(29)
        elif scales.ndim == 2:
            assert torch.equal(scales, scales[:1].expand_as(scales))
            scales = scales[0]
        if not native_vibe:
            agent['algorithm'].update(action_scales=scales.cpu().tolist(), action_joint_names=list(action.target_names))
        snapshot = copy.deepcopy(agent)
        runner = Runner(env, agent, str(out), device)
        if native_vibe:
            # Same tensor values, without object broadcast's GPU deserialization peak.
            from types import MethodType
            from ..da2_locomotion.algorithm import GeometryPPO
            runner.alg.broadcast_parameters = MethodType(GeometryPPO.broadcast_parameters, runner.alg)
        else:
            runner.alg.attach_geometry(collector)
        actor = runner.alg.get_policy()
        if dagger_env:
            dagger_env.attach(runner.alg, snapshot['actor'], snapshot['obs_groups'])
        motion = raw.command_manager.get_term('motion')
        terrain_hash = hashlib.sha256(json.dumps(motion.cfg.random_tiles, sort_keys=True).encode()).hexdigest()
        terrain_hashes = [terrain_hash]
        physical_hash=hashlib.sha256(json.dumps(motion.cfg.physical_tiles or motion.cfg.random_tiles,sort_keys=True).encode()).hexdigest()
        physical_hashes=[physical_hash]
        if world > 1:
            terrain_hashes = [None] * world
            torch.distributed.all_gather_object(terrain_hashes, terrain_hash)
            physical_hashes=[None]*world
            torch.distributed.all_gather_object(physical_hashes,physical_hash)
        initial_update = 0
        if args.resume:
            state = load_checkpoint(args.resume, map_location='cpu', weights_only=False)
            metadata = state['infos']['da2_stones']
            if args.migrate_domain:
                from .layout_randomization import validate_domain_migration
                validate_domain_migration(metadata['settings'],settings)
                if metadata['terrain_hashes']!=physical_hashes:
                    raise ValueError('Domain migration changed source physical mesh templates')
                record['domain_migration']=dict(source_checkpoint=str(Path(args.resume).resolve()),
                    retained='actor, critic, optimizer, valid geometry replay and curriculum frontier',
                    reset='curriculum rolling counts; fresh environments',
                    source_terrain_hashes=metadata['terrain_hashes'],layout_hashes=terrain_hashes,
                    change='per-episode colors/lights, layout descriptor bank, approach distance and local coordinates')
            if args.migrate_appearance:
                from .appearance import validate_appearance_migration
                validate_appearance_migration(metadata['settings'], settings)
                record['appearance_migration'] = dict(source_checkpoint=str(Path(args.resume).resolve()),
                    change='Episode-constant per-world floor/mat/pillar colors; optional validated local coordinates',
                    retained='actor, critic, optimizer, geometry replay and curriculum')
            if args.migrate_local_coordinates:
                from .local_coordinates import validate_coordinate_migration
                validate_coordinate_migration(metadata['settings'], settings)
                record['coordinate_migration'] = dict(version=1, source_checkpoint=str(Path(args.resume).resolve()),
                    retained='actor, critic, optimizer, geometry buffer and curriculum; fresh environment rollouts',
                    change='Per-world terrain translation with matching local robot/camera/analytic coordinates')
            if (metadata['settings'] != settings and not (args.migrate_local_coordinates or args.migrate_appearance or args.migrate_domain)) or metadata['world_size'] != world or (metadata['terrain_hashes'] != terrain_hashes and not args.migrate_domain):
                raise ValueError('Resume settings/world/terrain mismatch')
            runner.load(args.resume, map_location=device)
            if native_vibe:
                runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]['lr']
            initial_update = int(state['iter']) + 1
            runner.current_learning_iteration = initial_update
            auxiliary = load_checkpoint(Path(args.resume).parent / f'aux_{state["iter"]}_rank_{rank}.pt', map_location='cpu', weights_only=False)
            if dagger_env:
                dagger_env.load_state_dict(auxiliary[dagger_replay_key])
            collector.load_state_dict(auxiliary['geometry'])
            motion.restore_curriculum(auxiliary['curriculum'])
            record['resume'] = dict(checkpoint=str(Path(args.resume).resolve()), iteration=state['iter'],
                                    curriculum_frontier=motion.frontier,
                                    optimizer_lr=runner.alg.optimizer.param_groups[0]['lr'])
            if args.migrate_domain:
                motion.curriculum_counts.zero_();motion.recent_counts.zero_()
            record['initialization'] = metadata['stage1']
            raw.policy_update_count = initial_update
            raw.common_step_counter = initial_update * snapshot['num_steps_per_env']
            env.reset()
        elif args.init_actor and level10:
            source = Path(args.init_actor).resolve()
            state = load_checkpoint(source, map_location='cpu', weights_only=False)
            previous = state['infos']['da2_stones']['settings']
            def visual_contract(config):
                visual = dict(config['visual'])
                for key in ('source', 'checkpoint'):
                    visual[key] = str(Path(visual[key]).resolve())
                return visual
            if (not previous.get('stones_stairs_style') or previous.get('level10')
                    or previous['camera'] != settings['camera'] or visual_contract(previous) != visual_contract(settings)
                    or previous['stones']['minimum_support_fraction'] != settings['stones']['minimum_support_fraction']):
                raise ValueError('Expected a matching pretrained nine-pillar policy')
            it = int(state['iter'])
            sidecar = load_checkpoint(source.parent / f'aux_{it}_rank_{rank}.pt', map_location='cpu', weights_only=False)
            if int(sidecar['curriculum']['frontier']) != 9:
                raise ValueError('Source checkpoint has not reached level 9')
            actor.load_state_dict(state['actor_state_dict'], strict=True)
            del state, sidecar
            initial_update = it + 1
            runner.current_learning_iteration = initial_update
            raw.policy_update_count = initial_update
            raw.common_step_counter = initial_update * snapshot['num_steps_per_env']
            motion.frontier = 10
            motion.last_promotion = initial_update
            env.reset()
            record['initialization'] = dict(source_checkpoint=str(source), source_iteration=it,
                source_frontier=9, frontier=10, actor_loaded_strictly=True,
                fresh='critic, optimizer, replay buffers and curriculum statistics')
        elif args.init_actor:
            state = load_checkpoint(args.init_actor, map_location='cpu', weights_only=False)
            metadata = state['infos']['da2_stones']
            landing_changed = validate_actor_initialization(metadata['settings'], settings)
            if metadata['world_size'] != world or metadata['terrain_hashes'] != terrain_hashes:
                raise ValueError('Actor initialization changed world count or terrain')
            actor.load_state_dict(state['actor_state_dict'], strict=True)
            auxiliary = load_checkpoint(Path(args.init_actor).parent / f'aux_{state["iter"]}_rank_{rank}.pt',
                                        map_location='cpu', weights_only=False)
            collector.load_state_dict(auxiliary['geometry'])
            motion.restore_curriculum(auxiliary['curriculum'])
            if landing_changed:
                motion.curriculum_counts.zero_()
                motion.recent_counts.zero_()
            initial_update = int(state['iter']) + 1
            runner.current_learning_iteration = initial_update
            runner.alg.geometry_updates = state['stone_aux']['updates']
            runner.alg.gradient_audit = state['stone_aux']['gradient_audit']
            raw.policy_update_count = initial_update
            raw.common_step_counter = initial_update * snapshot['num_steps_per_env']
            env.reset()
            record['initialization'] = dict(source_checkpoint=str(Path(args.init_actor).resolve()),
                source_iteration=int(state['iter']), source_stage1=metadata['stage1'],
                landing_objective_changed=landing_changed,
                retained='Actor, geometry replay/warmup, curriculum frontier and appearance/layout schedule position',
                reset='Critic, optimizer and flat DAgger replay; fresh environment rollouts'
                      + ('; recent curriculum counts cleared for new landing objective' if landing_changed else ''))
        elif not raw_sonic:
            record['initialization'] = load_stage1(actor, settings['stage1_checkpoint'],
                                                  visual_backend=settings['visual']['backend'])
        if world > 1:
            runner.alg.broadcast_parameters()
        if raw_sonic and not (args.resume or args.init_actor):
            # Both visual architectures start from the same reset RNG and critic.
            torch.manual_seed(settings['seed'] + rank + 900000)
            env.reset()
            from ..stairs_ab.initialization import tensor_digest
            obs = env.get_observations()
            record['initialization'] = audit_raw_sonic(actor, obs[:16], snapshot, settings, device)
            record['initialization'].update(
                critic_initial_sha256=tensor_digest(runner.alg._raw_critic.state_dict().items()),
                initial_policy_observation_sha256=tensor_digest([('policy', obs['policy'])]),
                initial_tokenizer_observation_sha256=tensor_digest([('tokenizer', obs['tokenizer'])]))
        frozen_before = frozen_digest(actor)
        record['initial_parameter_sha256'] = shared_initial_digest(runner.alg)
        record['initial_distributed_audit'] = distributed_parameter_audit(runner.alg)
        record['terrain_ray_audit'] = audit_terrain_labels(raw)
        from .local_coordinates import audit_local_coordinates
        record['coordinate_audit'] = audit_local_coordinates(motion)
        if settings.get('appearance_randomization'):
            colors = raw.appearance_colors
            if not torch.isfinite(colors).all() or not torch.equal(colors[..., 3], torch.ones_like(colors[..., 3])):
                raise RuntimeError('Invalid randomized terrain RGBA')
            record['appearance_audit'] = dict(per_episode=True, physics_unchanged=True,
                nominal_world_fraction=float(raw.appearance_nominal.float().mean()),
                unique_pillar_colors=int(torch.unique(colors[:, 2, :3], dim=0).shape[0]))
        record['terrain_contract'] = audit_tiles(motion.cfg.random_tiles)
        if settings.get('layout_randomization'):
            record['layout_audit']=dict(descriptor_count=len(motion.cfg.random_tiles),
                physical_template_count=len(motion.cfg.physical_tiles),per_reset_sampling=True,
                variants_per_level=motion.cfg.profile_variants,approach_spawn_x_m=motion.cfg.approach_spawn_x)
        expected_dagger = (motion.flat_task if stairs_style else role_mask(motion.task_role, dagger_roles(standing))
                           if standing else torch.zeros(n, dtype=torch.bool, device=device))
        if (motion.paired.any() or not torch.equal(motion.dagger_mask, expected_dagger)
                or not torch.equal(motion.ppo_mask, ~expected_dagger)
                or not torch.equal(motion.ppo_online, ~expected_dagger)):
            raise RuntimeError('Unexpected PPO/task masks')
        split = dict(pillar_online_ppo=n, stairs=0, flat_replay=0, dagger=0)
        if settings.get('flat_training'):
            if n % 20:
                raise ValueError('Mixed task count must be divisible by 20')
            split = {name: int((motion.task_role == role).sum()) for role, name in enumerate(
                ('pillar_online_ppo', 'flat_standing_ppo', 'flat_walking_ppo', 'flat_start_stop_ppo'))}
            if standing:
                split['flat_standing_dagger'] = split.pop('flat_standing_ppo')
                if 2 in dagger_roles(standing):
                    split['flat_walking_dagger'] = split.pop('flat_walking_ppo')
                if 3 in dagger_roles(standing):
                    split['flat_start_stop_dagger'] = split.pop('flat_start_stop_ppo')
            split.update(stairs=0, flat_teacher_supervised=int((motion.flat_task & ~motion.dagger_mask).sum()))
            if standing:
                split.update(ppo_worlds=env.num_envs, flat_dagger_supervised=int(motion.dagger_mask.sum()))
            if not (motion.speed[motion.task_role == 1] == 0).all():
                raise RuntimeError('Standing task speed was overwritten')
        if stairs_style:
            from .stairs_style import task_audit
            split = task_audit(motion)
        obs = env.get_observations()
        if comparison:
            record['comparison_audit'] = audit_actor(actor, obs, settings)
        with torch.no_grad():
            sample = obs[:16]
            output = actor(sample, stochastic_output=False)
            modified = sample.clone()
            modified['flat_supervision'] = torch.ones_like(modified['flat_supervision']) * 42
            if settings.get('flat_training'):
                modified['flat_teacher_supervision'] = torch.ones_like(modified['flat_teacher_supervision']) * 42
            if not torch.equal(output, actor(modified, stochastic_output=False)):
                raise RuntimeError('Training-only supervision leaked into actor')
        record.update(status='running', frozen_sha256=frozen_before, started_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      stage2_start_iteration=initial_update, deployment_input_leak_test=True,
                      actual_task_split=split)
        write_json(status_path, record)
        if rank == 0:
            write_json(out / 'agent_cfg.json', snapshot)
            write_json(out / 'task_contract.json', dict(settings=settings,
                first_stage=record['initialization'], training_tasks=task_fractions(settings),
                actor_inputs=('Native Theia RGB patch/CLS, proprioception, ordinary SONIC reference and native 6D augmentation; no added command encoder'
                              if native_vibe else 'Native Theia RGB patch/CLS, proprioception, ordinary SONIC reference and speed; no pillar coordinates'
                              if matched_vibe else 'DA2 RGB history, proprioception, ordinary SONIC reference and speed; no pillar coordinates'),
                comparison=record.get('comparison_audit'),
                success=('Same-side first foot, alternate on every successive pillar after true swings; wrong-side landing or sustained crossover fails; supported upright far-floor traversal for 0.2s'
                         if settings['stones'].get('side_aware') else 'All pillars in order with alternating feet after true swings, then supported upright far-floor traversal for 0.2s'),
                start=('Original stairs rolling planner: persistent zero-speed stream or nominal 0.6 walking stream, no timed auto-start; replay uses original flat source clips'
                       if stairs_style else 'DAgger roles use original SONIC stand/continuous start-stop references and reference standing reset; other roles retain previous logic'
                       if standing else 'Flat stand/start-stop and 50% pillar episodes use zero root/joint velocities; half nominal stand, half reference stand; closed-loop deployment validation still required'
                       if settings.get('flat_training') else 'Walking-reference pose on ground, randomized phase; not standing-to-start deployment validation'),
                safe_flat_prior=('Disabled for native plain PPO' if native_vibe else 'Stairs loss and flat-neighborhood check; zero-speed included; circular edge gate 0.35..0.6m, first second 2x'
                                 if stairs_style else 'Approach/exit only; zero near pillars'),
                collision='64-sided convex round cylinders; maximum radial error <0.3 mm',
                schema_bank=('Complete per-rank bank; original flat replay sources are sampled for DAgger; no paired stairs PPO'
                             if stairs_style else 'Metadata only; mixed runs sample a separately generated continuous SONIC reference bank; no human replay or paired stairs'),
                flat_teacher=settings.get('dagger') if stairs_style else settings.get('flat_training'), actual_task_split=split,
                stones_stairs_style=settings.get('stones_stairs_style'),
                flat_standing_dagger=settings.get('flat_standing_dagger'),
                flat_dagger=settings.get('flat_dagger'),
                terrain_hashes=terrain_hashes))
            write_json(out / 'terrain_rank0.json', list(motion.cfg.random_tiles))
        old_update = runner.alg.update

        def monitored_update():
            if native_vibe:
                raw.da2_visual.encoder.assert_frozen()
            losses, info = old_update()
            if any(not math.isfinite(float(v)) for v in losses.values()):
                raise RuntimeError('Nonfinite PPO/auxiliary loss')
            completed = record['completed_updates'] + 1
            iteration = initial_update + completed - 1
            if stairs_style:
                record['actual_task_split'] = task_audit(motion)
            if level10:
                result = raw.route_results.detach()
                if world > 1:
                    result = result.clone()
                    torch.distributed.all_reduce(result)
                record['level10_routes'] = {name: dict(episodes=e, successes=s, failures=f)
                    for name, (e, s, f) in zip(('longitudinal', 'perpendicular', 'oblique'),
                                                result.cpu().tolist(), strict=True)}
            course = motion.update_curriculum(iteration + 1)
            record['latest_curriculum'] = course or record.get('latest_curriculum', {})
            metrics = {k: float(v / max(1, wrapped.metric_steps)) for k, v in wrapped.metric_sums.items()}
            wrapped.metric_sums.clear()
            wrapped.metric_steps = 0
            checkpoint_due = iteration % snapshot['save_interval'] == 0 or completed == iterations
            if checkpoint_due or (args.resume and completed == 1):
                record['distributed_audit'] = distributed_parameter_audit(runner.alg)
            if checkpoint_due:
                auxiliary = dict(geometry=collector.state_dict(), curriculum=motion.curriculum_state())
                if dagger_env:
                    auxiliary[dagger_replay_key] = dagger_env.state_dict()
                path = out / f'aux_{iteration}_rank_{rank}.pt'
                torch.save(auxiliary, str(path) + '.tmp')
                os.replace(str(path) + '.tmp', path)
                if world > 1:
                    torch.distributed.barrier()
            if native_vibe:
                # Release unused large minibatch buffers before the next Warp rollout.
                torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            record.update(status='running', completed_updates=completed, absolute_completed_updates=iteration + 1,
                updated_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), elapsed_s=time.monotonic() - start,
                loss=losses, metrics=metrics, curriculum_frontier=motion.frontier,
                episodes=raw.completed_episodes, strict_successes=raw.completed_successes, falls=raw.completed_falls,
                flat_episodes=raw.flat_episodes, flat_falls=raw.flat_falls,
                obstacle_idle_episodes=raw.obstacle_idle_episodes, obstacle_idle_falls=raw.obstacle_idle_falls,
                flat_stand_completed=raw.flat_stand_completed, flat_start_stop_completed=raw.flat_start_stop_completed,
                mean_terminal_pillar_fraction=raw.completed_progress_sum / max(1, raw.completed_episodes),
                device_used_gib=(total - free) / 2**30, gpu_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                geometry_gradient_audit=getattr(runner.alg, 'gradient_audit', None))
            write_json(status_path, record)
            with (out / f'rank_{rank}_metrics.jsonl').open('a') as f:
                f.write(json.dumps({k: record[k] for k in ('completed_updates', 'elapsed_s', 'loss', 'metrics',
                    'episodes', 'strict_successes', 'falls', 'curriculum_frontier', 'device_used_gib')}) + '\n')
            return losses, info

        runner.alg.update = monitored_update
        runner.learn(iterations, init_at_random_ep_len=False)
        raw.da2_visual.encoder.assert_frozen()
        if frozen_digest(actor) != frozen_before:
            raise RuntimeError('Frozen SONIC changed during training')
        record.update(status='completed', frozen_unchanged=True)
        write_json(status_path, record)
    except BaseException as exc:
        stack = traceback.format_exc()
        print(stack, flush=True)
        record.update(status='failed', error=repr(exc), traceback=stack)
        write_json(status_path, record)
        (out / f'rank_{rank}_traceback.txt').write_text(stack)
        if world > 1:
            import sys
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(1)
        raise
    finally:
        if env is not None:
            env.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
