"""DA2 visual conditioning with synchronized PPO/DAgger/terrain updates."""

from __future__ import annotations
from da2_vibe.paths import load_checkpoint
from da2_vibe.paths import load_settings, profile

import argparse
import copy
import json
import os
import time
import traceback
from pathlib import Path

from ..gpu_guard import verify_guarded_worker


def validate_mesh_resume(metadata, edge_hashes, world):
    """Each rank uses its own seeded terrain; compare the full rank-ordered set."""
    if metadata.get("world_size") != world:
        raise ValueError("Foothold resume world size changed")
    saved = metadata.get("edge_mesh_sha256_by_rank")
    if not isinstance(saved, list) or len(saved) != world:
        raise ValueError("Checkpoint lacks complete per-rank terrain mesh hashes")
    if saved != edge_hashes:
        raise ValueError("Hiking terrain mesh changed on resume")


def main(guard_check=verify_guarded_worker):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--num-envs", type=int)
    p.add_argument("--iterations", type=int)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--diagnostic", action="store_true", help="Bounded single-rank fault reproduction, not a full experiment")
    p.add_argument("--resume")
    p.add_argument("--migrate-render-domain", action="store_true")
    p.add_argument("--migrate-stair-failures", action="store_true")
    p.add_argument("--migrate-groot-visual", action="store_true")
    args = p.parse_args()
    if args.migrate_render_domain and not args.resume:
        raise ValueError('Render domain migration requires a complete resume checkpoint')
    if args.migrate_stair_failures and (not args.resume or args.migrate_render_domain):
        raise ValueError('Stair failure migration requires resume and cannot combine with render migration')
    if args.migrate_groot_visual and (not args.resume or args.migrate_render_domain or args.migrate_stair_failures):
        raise ValueError('GR00T visual migration requires resume and cannot combine with other migrations')
    guard_check()
    import mjlab  # noqa: F401 -- initialize upstream task discovery
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from ..catalog import write_json
    from ..common.checkpoint import frozen_digest
    from .task import agent_config, env_config, QualityStairsEnv, options, REMOVED_REWARDS, original_mixture, task_fractions

    settings = load_settings(args.config)
    if settings.get('comparison'):
        from ..stairs_ab.task import agent_config, env_config, QualityStairsEnv, options, REMOVED_REWARDS, original_mixture, task_fractions
    native_vibe = bool(settings.get('native_vibe'))
    if native_vibe:
        from ..stairs_ab.native_vibe import agent_config, env_config
    visual_contract = dict(camera=settings["camera"], visual=settings["visual"],
                           history_order="oldest_to_current", ground_truth_actor_input=False)
    fractions = task_fractions(settings)
    experiment_options = options(settings)
    if settings.get('stair_failures') and not args.resume:
        raise ValueError('Stair failure adaptation requires a complete resume checkpoint')
    if args.migrate_groot_visual:
        from ..da2_locomotion.groot_domain import validate_resume_options
    elif args.migrate_stair_failures:
        from ..velocity.stair_failure import validate_resume_options
    else:
        from ..da2_locomotion.render_domain import validate_resume_options
    migrate_options = args.migrate_stair_failures or args.migrate_render_domain or args.migrate_groot_visual
    dagger_enabled = settings.get("dagger") is not None
    stairs_only = settings.get('comparison', {}).get('stairs_only', False)
    sonic_checkpoint = (settings.get('dagger') or {}).get('teacher_checkpoint') or settings.get('sonic_checkpoint')
    if not dagger_enabled and not stairs_only:
        raise ValueError("Foothold comparison requires the existing DAgger split")
    if settings.get("speed_refinement") is not None and not args.resume:
        raise ValueError("Speed refinement requires --resume; actor-only warm start would reset the critic, optimizer and curriculum")
    rank, local, world = [
        int(os.environ.get(k, d)) for k, d in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"))
    ]
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    device = f"cuda:{local}"
    torch.manual_seed(settings["seed"] + rank)
    n = args.num_envs or settings["num_envs_per_rank"]
    iterations = args.iterations or settings["iterations"]
    if n < 10 or n % 10:
        raise ValueError("num-envs must be a multiple of 10")
    if dagger_enabled and (n < 20 or n % 20):
        raise ValueError("DAgger num-envs must be divisible by 20 for 50/20/15/15 fractions")
    if not args.smoke and not args.diagnostic and world != settings["expected_world_size"]:
        raise ValueError("Full experiment requires four merged data shards and four GPU ranks")
    if args.diagnostic and iterations > 200:
        raise ValueError("Fault reproduction is limited to 200 updates")
    from da2_vibe.ablations import single_gpu_ablation
    if world < 2 and not single_gpu_ablation(settings):
        raise ValueError("Single-GPU training requires an explicit stairs ablation contract")
    if args.smoke and iterations > 10:
        raise ValueError("Smoke is limited to 10 updates")
    bank = str(Path(settings["bank_root"]) / f"rank_{rank}")
    out = Path(args.log_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / f"rank_{rank}_status.json"
    if status_path.exists() and not args.resume:
        raise ValueError(f"Refusing to overwrite an existing run: {out}")
    record = {
        "status": "initializing",
        "rank": rank,
        "world_size": world,
        "num_envs": n,
        "settings": settings,
        "requested_iterations": iterations,
        "smoke": args.smoke,
        "diagnostic": args.diagnostic,
        "initialization": args.resume or settings.get('stage1_checkpoint') or settings.get('repair_warm_start') or sonic_checkpoint,
        "visual_contract": visual_contract,
    }
    write_json(status_path, record)
    started = time.monotonic()

    class LoggingWrapper(RslRlVecEnvWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.returns = torch.zeros(n, device=device)
            self.lengths = torch.zeros(n, device=device)
            self.counters = {"online_episodes": 0, "online_success": 0, "online_falls": 0}
            if settings.get('stair_failures'):
                self.counters.update(stair_edge_failures=0, foot_collision_failures=0,
                                     online_failures=0, online_timeouts=0)
            for name in ("up", "down", "up_down", "online_flat"):
                for suffix in ("episodes", "success", "falls"):
                    self.counters[f"{name}_{suffix}"] = 0
            for level in range(settings.get("profile_levels", 8)):
                for suffix in ("episodes", "success", "falls"):
                    self.counters[f"level_{level}_{suffix}"] = 0
            self.recent = {}
            self.posture_sums = {}
            self.posture_steps = 0
            self.quality_sums = {}
            self.quality_steps = 0
            self.failure_examples = 0

        def step(self, actions):
            c = self.unwrapped.command_manager.get_term("motion")
            online_mask = getattr(c, "ppo_online", c.online)
            posture_metrics = {}
            if settings.get("posture_revision") is not None:
                from ..velocity.posture_task import posture_scalar_metrics

                posture_metrics = {f"Posture/{k}": v for k, v in posture_scalar_metrics(self.unwrapped).items()}
                for key, value in posture_metrics.items():
                    self.posture_sums[key] = self.posture_sums.get(key, 0.) + value.detach()
                self.posture_steps += 1
            # Before step: these are genuine states, never auto-reset positions.
            vel = c.robot.data.root_link_lin_vel_w
            error = ((vel[:, 0] - c.speed).square() + vel[:, 1].square()).sqrt()
            progress = c.robot.data.root_link_pos_w[:, 0] - c.tile_origins[c.tile, 0]
            terrain_kind = c.tile_data["kind"][c.tile].clone()
            terrain_level = c.tile_data["level"][c.tile].clone()
            moving = c.speed > 0.1
            obs, reward, done, extras = super().step(actions)
            if settings.get('stair_failures'):
                # True failure overrides simultaneous time limit/success bootstrap.
                if 'time_outs' in extras:
                    extras['time_outs'] = extras['time_outs'] & ~self.unwrapped.reset_terminated
            self.returns += reward
            self.lengths += 1
            logs = extras.setdefault("log", {})
            logs.update(posture_metrics)
            logs.update(self.unwrapped.guidance.metrics)
            if settings.get('stair_failures'):
                logs.update(self.unwrapped.stair_failures.metrics)
            for key, value in self.unwrapped.guidance.metrics.items():
                if key.startswith(("StepQuality/", "Hiking/", "SSR/", "FootPose/")):
                    self.quality_sums[key] = self.quality_sums.get(key, 0.) + (value.detach() if torch.is_tensor(value) else value)
            if settings.get('stair_failures'):
                for key, value in self.unwrapped.stair_failures.metrics.items():
                    self.quality_sums[key] = self.quality_sums.get(key, 0.) + (value.detach() if torch.is_tensor(value) else value)
            self.quality_steps += 1
            for name in ("online", "paired", "flat"):
                mask = online_mask if name == "online" else getattr(c, name)
                logs[f"Velocity/{name}_reward"] = reward[mask].mean() if mask.any() else 0.0
                ended = mask & done.bool()
                if ended.any():
                    logs[f"Velocity/{name}_episode_return"] = self.returns[ended].mean()
                    logs[f"Velocity/{name}_episode_seconds"] = self.lengths[ended].mean() * 0.02
            ended = online_mask & done.bool()
            success = c.terminal_success & ended
            fallen = c.terminal_fall & ended
            if settings.get('stair_failures'):
                edge = c.terminal_stair_edge & ended
                pair = c.terminal_foot_collision & ended
                failed = fallen | edge | pair
                self.counters['stair_edge_failures'] += int(edge.sum())
                self.counters['foot_collision_failures'] += int(pair.sum())
                self.counters['online_failures'] += int(failed.sum())
                self.counters['online_timeouts'] += int((ended & ~success & ~failed).sum())
                # Bound diagnostic IO; use the pre-reset snapshot, never fresh spawn poses.
                if self.failure_examples < 200 and (edge | pair).any():
                    observer = self.unwrapped.stair_failures
                    snap = observer.snapshot
                    selected = (edge | pair)[observer.ids].nonzero().flatten()
                    with (out / f'rank_{rank}_failure_events.jsonl').open('a') as file:
                        for i in selected[:min(8, 200-self.failure_examples)].tolist():
                            world_id = int(observer.ids[i])
                            row = dict(absolute_update=int(getattr(self.unwrapped, 'policy_update_count', 0)),
                                step=int(self.unwrapped.common_step_counter), env=world_id,
                                edge=bool(edge[world_id]), pair=bool(pair[world_id]),
                                fall=bool(fallen[world_id]), tile=int(snap['tile'][i]),
                                margin_m=snap['margin'][i].cpu().tolist(), load_n=snap['load'][i].cpu().tolist(),
                                pair_peak_n=float(snap['pair_peak'][i]), qpos=snap['qpos'][i].cpu().tolist(),
                                root=snap['root'][i].cpu().tolist(), threshold=snap['threshold'])
                            file.write(json.dumps(row)+'\n')
                            self.failure_examples += 1
                if ended.any():
                    logs['Velocity/online_failure_rate'] = failed.sum().float() / ended.sum()
                    logs['StairFailure/edge_episode_rate'] = edge.sum().float() / ended.sum()
                    logs['StairFailure/foot_collision_episode_rate'] = pair.sum().float() / ended.sum()
            self.counters["online_episodes"] += int(ended.sum())
            self.counters["online_success"] += int(success.sum())
            self.counters["online_falls"] += int(fallen.sum())
            for kind, name in enumerate(("up", "down", "up_down", "online_flat")):
                terminal = ended & moving & (terrain_kind == kind)
                self.counters[f"{name}_episodes"] += int(terminal.sum())
                self.counters[f"{name}_success"] += int((terminal & success).sum())
                self.counters[f"{name}_falls"] += int((terminal & fallen).sum())
                if settings.get('stair_failures'):
                    for reason, flag in (('failures', failed), ('edge', edge), ('foot_collision', pair)):
                        key = f'{name}_{reason}'
                        self.counters[key] = self.counters.get(key, 0) + int((terminal & flag).sum())
                if terminal.any():
                    logs[f"Velocity/{name}_success_rate"] = (
                        terminal & success
                    ).sum().float() / terminal.sum()
            for level in range(settings.get("profile_levels", 8)):
                terminal = ended & moving & (terrain_kind != 3) & (terrain_level == level)
                self.counters[f"level_{level}_episodes"] += int(terminal.sum())
                self.counters[f"level_{level}_success"] += int((terminal & success).sum())
                self.counters[f"level_{level}_falls"] += int((terminal & fallen).sum())
                if settings.get('stair_failures'):
                    for reason, flag in (('failures', failed), ('edge', edge), ('foot_collision', pair)):
                        key = f'level_{level}_{reason}'
                        self.counters[key] = self.counters.get(key, 0) + int((terminal & flag).sum())
            logs["Velocity/online_speed_error_m_s"] = error[online_mask].mean()
            logs["Velocity/online_progress_m"] = progress[online_mask].mean()
            logs["Velocity/terrain_level"] = c.tile_data["level"][c.tile[online_mask]].float().mean()
            logs["Velocity/planner_calls"] = float(c.pool.calls)
            logs["Velocity/gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            if ended.any():
                logs["Velocity/online_success_rate"] = success.sum().float() / ended.sum()
                logs["Velocity/online_fall_rate"] = fallen.sum().float() / ended.sum()
            self.recent = {k: float(v) for k, v in logs.items()
                           if k.startswith(("Velocity/", "SSR/", "Hiking/", "FootPose/", "StepQuality/", "StairFailure/"))}
            self.returns[done.bool()], self.lengths[done.bool()] = 0, 0
            return obs, reward, done, extras

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

            configure_distributed(self, settings.get("collective_timeout_seconds", 180))

        def save(self, path, infos=None):
            # Atomic checkpoint writes keep the latest complete file reviewable.
            if frozen_digest(self.alg.get_policy()) != before:
                raise RuntimeError("Frozen SONIC parameters changed before checkpoint save")
            tmp = str(path) + ".tmp"
            from ..velocity.recovery_support import checkpoint_metadata

            metadata = checkpoint_metadata(self, settings, infos)
            metadata["velocity_training"].update(replay_fraction=fractions['paired_grail_ppo'], online_fraction=fractions['online_ppo'],
                dagger_fraction=fractions['flat_dagger_planner'] + fractions['flat_dagger_replay'])
            if dagger_enabled:
                metadata["dagger_replay"] = dict(world_size=world, num_envs_per_rank=n,
                                                 pattern=f"dagger_{self.current_learning_iteration}_rank_{{rank}}.pt")
            metadata["da2_locomotion"] = dict(version=1, settings=settings["visual"],
                ascent_only=False, task_mode=settings.get('task_mode', 'ascent_only_v1'),
                fractions=fractions, independent_geometry_branch=False, camera=settings["camera"])
            metadata["foothold_experiment"] = dict(version=2, options=experiment_options, world_size=world,
                edge_mesh_sha256=env.unwrapped.guidance.edge_hash,
                edge_mesh_sha256_by_rank=edge_hashes)
            super().save(tmp, metadata)
            os.replace(tmp, path)

    env = None
    try:
        cfg = env_config(bank, settings, n)
        # Establish the absolute update before the first resumed episode reset.
        resume_updates = 0
        if args.resume:
            header = load_checkpoint(args.resume, map_location="cpu", weights_only=False)
            if settings.get('stair_failures'):
                if args.migrate_groot_visual:
                    from ..da2_locomotion.groot_domain import validate_resume_options as validate_stair_settings
                else:
                    from ..velocity.stair_failure import validate_resume_options as validate_stair_settings
                validate_stair_settings(header.get('infos', {}).get('velocity_training', {}).get('settings'),
                                        settings, args.migrate_stair_failures or args.migrate_groot_visual)
            validate_resume_options(header.get("infos", {}).get("foothold_experiment", {}).get("options"),
                                    experiment_options, migrate_options)
            resume_updates = int(header["iter"]) + 1
            if args.migrate_stair_failures and settings['stair_failures']['start_update'] != resume_updates:
                raise ValueError('Stair failure adaptation must start at this checkpoint update')
            del header
        agent = agent_config(bank, settings, n)
        if args.smoke:
            agent["save_interval"] = 1
        from ..common.cuda_graph import build_sensor_env
        raw_env = build_sensor_env(QualityStairsEnv, cfg, device=device, settings=settings,
            disable_graph_capture=profile().get('runtime', {}).get('disable_cuda_graph_capture', False))
        logging_env = LoggingWrapper(raw_env)
        from ..da2_locomotion.camera import camera_contract, assert_sim_camera
        record['actual_camera'] = assert_sim_camera(logging_env.unwrapped,
            camera_contract(next(s for s in cfg.scene.sensors if s.name == 'head_cam')))
        from ..da2_locomotion.collection import GeometryCollector
        collector = GeometryCollector(logging_env, settings["geometry_aux"], rank, enabled=not native_vibe)
        env = collector
        if dagger_enabled:
            from ..da2_locomotion.collection import GeometryDaggerEnv as DaggerVecEnv

            env = DaggerVecEnv(env, settings, rank)
        if settings.get("posture_revision") is not None and not native_vibe:
            action = env.unwrapped.action_manager.get_term("joint_pos")
            names = list(action.target_names)
            if names != list(env.unwrapped.scene["robot"].joint_names):
                raise RuntimeError("SONIC action order differs from robot joint order")
            scale = torch.as_tensor(action.scale, device=device)
            if scale.ndim == 0:
                scale = scale.expand(29)
            elif scale.ndim == 2:
                if not torch.equal(scale, scale[:1].expand_as(scale)):
                    raise RuntimeError("Upper action prior requires the same target scales in every world")
                scale = scale[0]
            agent["algorithm"].update(action_scales=scale.cpu().tolist(), action_joint_names=names)
        agent["actor"]["base_checkpoint"] = sonic_checkpoint
        agent_snapshot = copy.deepcopy(agent)
        runner = Runner(env, agent, str(out), device)
        edge_hashes = [env.unwrapped.guidance.edge_hash]
        if world > 1:
            edge_hashes = [None] * world
            torch.distributed.all_gather_object(edge_hashes, env.unwrapped.guidance.edge_hash)
        env.unwrapped.guidance.synchronize_initial_model()
        from ..da2_locomotion.collection import audit_terrain_labels
        record['terrain_label_audit'] = audit_terrain_labels(env.unwrapped)
        actor = runner.alg.get_policy()
        if not native_vibe:
            runner.alg.attach_geometry(collector)
        if dagger_enabled:
            env.attach(runner.alg, agent_snapshot["actor"], agent_snapshot["obs_groups"])
        elif stairs_only:
            from ..velocity.dagger_policy import build_teacher
            audit_teacher, teacher_sha = build_teacher(env.get_observations()[:8], agent_snapshot['actor'],
                agent_snapshot['obs_groups'], sonic_checkpoint, device)
            runner.alg.teacher_sha256 = teacher_sha
        from ..stairs_ab.ablation import reset_with_ablation_audit
        if args.resume:
            restored_infos = runner.load(args.resume, map_location=device)
            if dagger_enabled:
                from ..velocity.dagger_checkpoint import restore_replay

                record["dagger_replay_restore"] = restore_replay(
                    env, args.resume, runner.current_learning_iteration, rank, world, n,
                    runner.alg.teacher_sha256, restored_infos,
                    allow_env_count_change=settings["dagger"].get("allow_env_count_change_on_resume", False),
                )
            validate_resume_options(restored_infos.get("foothold_experiment", {}).get("options"),
                                    experiment_options, migrate_options)
            validate_mesh_resume(restored_infos["foothold_experiment"], edge_hashes, world)
            env.unwrapped.guidance.load(args.resume, runner.current_learning_iteration, rank, world)
            aux_path = Path(args.resume).parent / f"geometry_{runner.current_learning_iteration}_rank_{rank}.pt"
            collector.load_state_dict(load_checkpoint(aux_path, map_location='cpu', weights_only=False))
            runner.current_learning_iteration += 1
            # Mjlab restores common_step_counter; also synchronize the cached
            # curriculum value before resetting the newly created physics worlds.
            env.unwrapped.common_step_counter = resume_updates * agent_snapshot["num_steps_per_env"]
            env.unwrapped.policy_update_count = resume_updates
            env.unwrapped.command_manager.get_term("motion")._init_phase_max = max(
                0.0, 1.0 - resume_updates / max(1, settings["replay_phase_anneal_iterations"])
            )
            reset_with_ablation_audit(env, settings, record)
        elif settings.get('initialization') == 'raw_sonic_fresh_visual':
            if not settings.get('comparison'):
                raise ValueError('Raw SONIC mode requires an explicit matched comparison config')
            runner.current_learning_iteration = 0
            runner.alg.broadcast_parameters()
            # Restore a shared reset RNG after method-specific model construction.
            torch.manual_seed(settings['seed'] + rank + 900000)
            reset_with_ablation_audit(env, settings, record)
            from ..stairs_ab.initialization import audit_raw_initialization, tensor_digest
            sample = env._full_obs[:8] if dagger_enabled else env.get_observations()[:8]
            record['raw_sonic_initialization'] = audit_raw_initialization(actor,
                env.teacher if dagger_enabled else audit_teacher, sample)
            record['raw_sonic_initialization'].update(
                checkpoint=sonic_checkpoint, checkpoint_sha256=runner.alg.teacher_sha256,
                critic_initial_sha256=tensor_digest(runner.alg._raw_critic.state_dict().items()),
                initial_policy_observation_sha256=tensor_digest([('policy', env.get_observations()['policy'])]),
            )
            if native_vibe:
                record['raw_sonic_initialization']['fresh_command_encoder'] = False
                record['raw_sonic_initialization']['native_6d_command_stream'] = True
            if stairs_only:
                del audit_teacher
        elif settings.get('initialization') == 'stage1_nonvisual_fresh_da2':
            from ..da2_stones.initialization import load_stage1
            record['fresh_stage2'] = load_stage1(actor, settings['stage1_checkpoint'])
            runner.current_learning_iteration = 0
            runner.alg.broadcast_parameters()
        else:
            # Deliberate fine-tune: retain actor and critic, reset optimizer/replays for changed losses.
            state = load_checkpoint(settings['repair_warm_start'], map_location=device, weights_only=False)
            actor.load_state_dict(state['actor_state_dict'], strict=True)
            runner.alg._raw_critic.load_state_dict(state['critic_state_dict'], strict=True)
            resume_updates = int(state['iter']) + 1
            runner.current_learning_iteration = resume_updates
            env.unwrapped.common_step_counter = resume_updates * agent_snapshot['num_steps_per_env']
            env.unwrapped.policy_update_count = resume_updates
            env.unwrapped.command_manager.get_term('motion')._init_phase_max = max(
                0., 1. - resume_updates / settings['replay_phase_anneal_iterations'])
            reset_with_ablation_audit(env, settings, record)
            record['fresh_stage2'] = dict(initialization='DA2 actor and critic warm start',
                checkpoint=settings['repair_warm_start'], source_iteration=int(state['iter']),
                fresh_optimizer=True, fresh_replay_buffers=True, unchanged_actor_architecture=True)
            del state
        before = frozen_digest(actor)
        no_geometry_ablation = settings.get('ablation', {}).get('variant') in ('no_geometry', 'blind')
        blind_ablation = settings.get('ablation', {}).get('variant') == 'blind'
        if blind_ablation:
            from ..stairs_ab.ablation import audit_blind_observations
        if no_geometry_ablation:
            from ..stairs_ab.initialization import tensor_digest
            geometry_head_before = tensor_digest(actor.terrain_head.state_dict().items())
            record['no_geometry_audit'] = dict(
                initial_head_sha256=geometry_head_before, coefficient=0.,
                private_sampling_rng_preserved=True)
        from ..common.audit import shared_initial_digest
        record['initial_shared_actor_and_critic_sha256'] = shared_initial_digest(runner.alg)
        if dagger_enabled:
            with torch.no_grad():
                sample = env._full_obs[:min(n, 8)]
                error = (env.teacher(sample) - actor.sonic_action(sample)).abs().max()
                if float(error) > 1e-5:
                    raise RuntimeError(f"Student frozen base differs from original SONIC teacher: {float(error)}")
            record["teacher_base_max_error"] = float(error)
        record.update(
            status="running",
            render_domain=settings.get('render_domain'),
            render_domain_migrated=args.migrate_render_domain,
            stair_failures=settings.get('stair_failures'),
            stair_failures_migrated=args.migrate_stair_failures,
            groot_visual=settings.get('groot_visual'),
            groot_visual_migrated=args.migrate_groot_visual,
            frozen_sha256=before,
            total_actor_parameters=sum(p.numel() for p in actor.parameters()),
            trainable_actor_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad),
            actor_groups=list(env.get_observations().keys()),
            resumed_after_updates=resume_updates,
            started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        write_json(status_path, record)
        if rank == 0:
            (out / "agent_cfg.json").write_text(json.dumps(agent_snapshot, indent=2, default=str))
            write_json(
                out / "task_contract.json",
                {
                    "robot": "G1 29DOF",
                    "actor_camera": settings["camera"],
                    "actual_camera_fovy_degrees": next(s.fovy for s in cfg.scene.sensors if s.name == 'head_cam'),
                    "visual_contract": visual_contract,
                    "visual_encoder": settings["visual"],
                    "geometry_aux": settings["geometry_aux"],
                    "ablation": settings.get('ablation'),
                    "visual_encoder_deployment_required": settings['visual'].get('ablation_input') != 'constant_tokens',
                    "independent_geometry_branch": False,
                    "ascent_only": not original_mixture(settings),
                    "task_mode": settings.get('task_mode', 'ascent_only_v1'),
                    "depth_input": False,
                    "base": ("frozen original SONIC; native ViBe actor, fresh LoRA and Theia fusion, native 6D augmentation; plain native PPO, no preservation or auxiliary objectives"
                             if native_vibe else "frozen original SONIC; fresh method-specific visual fusion, LoRA and shared command branch; initialization in raw_sonic_initialization"
                             if settings.get('comparison') else "frozen SONIC and DA2; train query multi-head fusion, LoRA, command encoder and terrain head; initialization in fresh_stage2"),
                    "online_reference": "runtime CPU ONNX, 5Hz, five shared rolling streams per rank; independent physics, phase and terrain",
                    "command": "independent target planar speed and +X heading, velocity_command[8]; stop uses planner idle mode",
                    "fractions": fractions,
                    "dagger": settings.get("dagger"),
                    "foothold_online_ppo_only": experiment_options,
                    "removed_rewards": list(REMOVED_REWARDS),
                    "quality_revision": settings["step_quality"],
                    "stair_failures": settings.get('stair_failures'),
                    "slip_measurement": "force-gated actual contact-point tangential velocity at every physics substep",
                    "contact_point_slip_weight": cfg.rewards["online_slip"].weight,
                    "auxiliary_deployment_required": False,
                    "ssr_paper": "https://arxiv.org/html/2605.30770v1#S3.SS2",
                    "method": settings["foothold_experiment"]["method"],
                    "hiking_paper": "https://arxiv.org/html/2601.07718v1#S3.SS3",
                    "edge_mesh_sha256": env.unwrapped.guidance.edge_hash,
                    "edge_mesh_sha256_by_rank": edge_hashes,
                    "edge_counts": env.unwrapped.guidance.edge_counts,
                    "teacher_sha256": getattr(runner.alg, "teacher_sha256", None),
                    "privileged": "critic/rewards and training-only terrain ray-hit targets; actor forward receives RGB features, proprioception, reference and command only",
                    "curriculum": ("normal stairs: 2.5-25cm risers, 4-8 steps, 28-40cm treads independent of riser difficulty; " if settings.get("terrain_profile") else "4-18cm uniform stair risers; ") + f"starts at first two levels, adds level every {settings['curriculum_interval']} updates",
                    "scope": ("100% stair PPO (up/down/up_down), all using flat planner references; zero independent flat environments, zero DAgger, zero replay-reference PPO"
                              if stairs_only else "70% online PPO with flat planner reference on up/down/up_down/flat; 15% flat planner DAgger; 15% flat replay DAgger; no paired GRAIL PPO"
                              if settings.get('comparison') else "original 50/20/15/15 mixture: online up/down/up_down/flat, paired GRAIL PPO, both flat DAgger sources"
                              if original_mixture(settings) else "archived ascent-only task"),
                    "frozen_sha256": before,
                    "posture_revision": settings.get("posture_revision"),
                    "posture_metrics": "pre-step online measurements averaged across the complete rollout" if settings.get("posture_revision") is not None else None,
                    "initialization_mode": "restore model, critic and optimizer" if args.resume else settings['initialization'],
                    "fresh_stage2": record.get('fresh_stage2'),
                    "raw_sonic_initialization": record.get('raw_sonic_initialization'),
                    "comparison": settings.get('comparison'),
                    "event_names": list(cfg.events),
                    "termination_names": list(cfg.terminations),
                    "actual_camera": record['actual_camera'],
                },
            )
        # Record all ranks after every update, not just rank-zero console output.
        if args.diagnostic and native_vibe:
            from ..stairs_ab.numeric_probe import install_probe
            install_probe(runner.alg, out, rank, resume_updates)
        original_update = runner.alg.update

        def monitored_update():
            losses, info = original_update()
            if blind_ablation and (env.unwrapped.da2_visual.calls or env.unwrapped.da2_visual.images):
                raise RuntimeError('Blind ablation unexpectedly encoded an image')
            if no_geometry_ablation:
                if (runner.alg._aux_count != 0 or runner.alg.gradient_audit is not None
                        or any(p.grad is not None for p in actor.terrain_head.parameters())):
                    raise RuntimeError('No-geometry ablation unexpectedly received auxiliary gradients')
            losses.update(env.unwrapped.guidance.update())
            from ..common.audit import distributed_parameter_audit, task_audit
            record['actual_task_split'] = task_audit(env.unwrapped.command_manager.get_term('motion'),
                'stairs_only_reference_ab_v1' if stairs_only else
                'stairs_flat_reference_ab_v1' if settings.get('comparison') else settings.get('task_mode'))
            import math
            if any(not math.isfinite(float(value)) for value in losses.values()):
                raise RuntimeError(f"Nonfinite training losses: {losses}")
            # All ranks save geometry/guidance sidecars, including PPO-only A/B.
            it = resume_updates + record.get("completed_updates", 0)
            if it % agent_snapshot["save_interval"] == 0 or it == resume_updates + iterations - 1:
                if blind_ablation:
                    record['blind_input_audit'] = audit_blind_observations(
                        env.unwrapped, collector.replay['obs'])
                    record['blind_input_audit']['checked_iteration'] = it
                if no_geometry_ablation:
                    head_after = tensor_digest(actor.terrain_head.state_dict().items())
                    if head_after != geometry_head_before:
                        raise RuntimeError('No-geometry ablation changed the unused terrain head')
                    record['no_geometry_audit'].update(
                        checked_iteration=it, head_sha256=head_after,
                        head_unchanged=True, auxiliary_backward_calls=0)
                record['distributed_parameter_audit'] = distributed_parameter_audit(runner.alg)
                if dagger_enabled:
                    from ..velocity.dagger_checkpoint import save_replay
                    save_replay(env, out, it, rank, world, n, runner.alg.teacher_sha256)
                env.unwrapped.guidance.save(out, it, rank, world)
                aux_path = out / f"geometry_{it}_rank_{rank}.pt"
                torch.save(collector.state_dict(), str(aux_path) + '.tmp')
                os.replace(str(aux_path) + '.tmp', aux_path)
                if world > 1: torch.distributed.barrier()
            rollout_metrics = dict(env.recent)
            rollout_metrics.update({k: float(v) for k, v in info.items() if k.startswith(("Geometry/", "FlatRepair/"))})
            if env.quality_steps:
                rollout_metrics.update({k: float(v / env.quality_steps) for k, v in env.quality_sums.items()})
                env.quality_sums.clear()
                env.quality_steps = 0
            rollout_metrics.update(env.unwrapped.guidance.update_metrics())
            if settings.get('groot_visual'):
                visual_state=env.unwrapped.scene.sensor_context.render_context.groot_state
                rgb_parameters=visual_state['rgb']
                rollout_metrics.update({
                    'VisualDR/hdr_resamples':float(visual_state['reset_count']),
                    'VisualDR/episode_samples':float(visual_state['episode_count']),
                    'VisualDR/hdr_index':float(visual_state.get('hdr_index',-1)),
                    'VisualDR/hdr_intensity':float(visual_state.get('hdr_intensity',0)),
                    'VisualDR/rgb_enabled':float(visual_state['active']),
                    'VisualDR/brightness_fraction':float((rgb_parameters[:,0]!=1).float().mean()),
                    'VisualDR/hue_fraction':float((rgb_parameters[:,3]!=0).float().mean()),
                    'VisualDR/blur_fraction':float(rgb_parameters[:,5].mean()),
                    'VisualDR/noise_fraction':float((rgb_parameters[:,4]>0).float().mean()),
                })
            if env.posture_steps:
                rollout_metrics.update({k: float(v / env.posture_steps) for k, v in env.posture_sums.items()})
                env.posture_sums.clear()
                env.posture_steps = 0
                rollout_metrics.update({f"PostureAction/{k}": float(v) for k, v in info.items()
                                        if k.startswith("online_")})
            motion = env.unwrapped.command_manager.get_term("motion")
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_gib = (total_bytes - free_bytes) / 2**30
            record.update(
                status="running",
                completed_updates=record.get("completed_updates", 0) + 1,
                absolute_completed_updates=resume_updates + record.get("completed_updates", 0) + 1,
                policy_update_count=int(getattr(env.unwrapped, "policy_update_count", 0)),
                replay_init_phase_max=float(motion._init_phase_max),
                learning_rate=float(runner.alg.learning_rate),
                updated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                elapsed_s=time.monotonic() - started,
                loss=losses,
                metrics=rollout_metrics,
                counters=env.counters,
                encoder_calls=env.unwrapped.da2_visual.calls,
                encoded_images=env.unwrapped.da2_visual.images,
                gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                gpu_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                device_used_gib=used_gib,
                max_observed_device_used_gib=max(record.get("max_observed_device_used_gib", 0.0), used_gib),
                ppo_envs=env.num_envs, dagger_envs=len(env.dagger_ids) if dagger_enabled else 0,
                flat_replay_data_used=dagger_enabled,
            )
            write_json(status_path, record)
            with (out / f"rank_{rank}_metrics.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {
                            k: record[k]
                            for k in (
                                "completed_updates",
                                "absolute_completed_updates",
                                "elapsed_s",
                                "loss",
                                "metrics",
                                "counters",
                                "gpu_peak_reserved_gib",
                            )
                        }
                    )
                    + "\n"
                )
            if native_vibe:
                # Native 4-mini-batch PPO has a much larger transient allocator
                # peak than rollout physics. Return unused PyTorch blocks to CUDA
                # before Warp's next graph launch requests its own workspace.
                torch.cuda.empty_cache()
            return losses, info

        runner.alg.update = monitored_update
        runner.learn(iterations, init_at_random_ep_len=False)
        env.unwrapped.da2_visual.encoder.assert_frozen()
        after = frozen_digest(actor)
        if before != after:
            raise RuntimeError("Frozen SONIC weights changed")
        if dagger_enabled:
            if env.teacher.training or any(p.requires_grad or p.grad is not None for p in env.teacher.parameters()):
                raise RuntimeError("DAgger teacher must remain frozen and in eval mode")
            record.update(teacher_sha256=runner.alg.teacher_sha256, teacher_frozen=True,
                          ppo_envs=env.num_envs, dagger_envs=len(env.dagger_ids),
                          dagger_history_samples=len(env.buffer))
        record.update(status="completed", frozen_unchanged=True, frozen_sha256_after=after)
        write_json(status_path, record)
    except BaseException as e:
        # Write the ORIGINAL stack before cleanup can block on another NCCL rank.
        stack = traceback.format_exc()
        print(stack, flush=True)
        (out / f"rank_{rank}_traceback.txt").write_text(stack)
        record.update(status="failed", error=repr(e), traceback=stack)
        write_json(status_path, record)
        if world > 1:
            # A failing rank must not wait in CUDA/NCCL teardown while its peers
            # are blocked in a collective. torchrun will terminate our sibling ranks.
            import sys

            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        raise
    finally:
        if env is not None:
            env.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
