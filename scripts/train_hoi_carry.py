"""Train the visual whole-body tracker on one physical G1 bimanual carry."""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
from types import MethodType

import _bootstrap
import torch
import numpy as np
from da2_vibe.paths import load_settings, load_checkpoint
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from orcs.core.rl import runner as ppo_config
from vr_vibe.da2_task import agent_config
from hoi_training import Runner, native_broadcast, write_json
from hoi_carry_task import config


def main(a):
    rank = int(os.getenv("LOCAL_RANK", 0))
    world = int(os.getenv("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    device = f"cuda:{rank}"
    seed = 20261124 + rank
    torch.manual_seed(seed)
    a.output.mkdir(parents=True, exist_ok=True)
    settings = load_settings(str(_bootstrap.ROOT / "configs/tasks/teleop.json"))
    if a.resume:
        saved = load_checkpoint(a.resume, map_location="cpu", weights_only=False)
        version = int(saved["actor_state_dict"].get("hoi_reference_version", 0))
        if a.init is None:
            a.init = "sonic" if version else "legacy"
        if (a.init == "sonic") != bool(version) or version not in (0, 1, 2, 3, 4, 5):
            raise ValueError("Checkpoint is incompatible with the requested architecture")
        if ((a.grasp_feedback or a.smooth_grasp or a.visual_pickup) and version not in (2, 4, 5)) or (
            a.full_body and version != 3
        ):
            raise ValueError("Requested controller differs from the resumed checkpoint")
        a.grasp_feedback = version in (2, 4, 5)
        a.visual_pickup = a.visual_pickup or version == 5
        a.smooth_grasp = a.smooth_grasp or version in (4, 5)
        a.full_body = version == 3
        del saved
    a.smooth_grasp = a.smooth_grasp or a.visual_pickup
    a.grasp_feedback = a.grasp_feedback or a.smooth_grasp
    original = (a.init or "sonic") == "sonic"
    checkpoint = (
        settings["dagger"]["teacher_checkpoint"]
        if original
        else load_settings(str(_bootstrap.ROOT / "configs/demos/revo2_pick_carry.json"))["checkpoint"]
    )
    if (a.grasp_feedback or a.full_body) and not original:
        raise ValueError("Reference-conditioned controllers require the original SONIC architecture")
    cfg = config(
        a.bundle,
        settings,
        a.num_envs,
        seed,
        reference_tracking=original,
        grasp_feedback=a.grasp_feedback,
        full_body=a.full_body,
        smooth_grasp=a.smooth_grasp,
        visual_pickup=a.visual_pickup,
    )
    env = ManagerBasedRlEnv(cfg, device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    cmd = env.command_manager.get_term("motion")
    with np.load(a.bundle / "motion.npz") as ref:
        assert tuple(cmd.robot.joint_names) == tuple(ref["joint_names"])
        assert tuple(cmd.robot.body_names) == tuple(ref["body_names"])
    ag = asdict(ppo_config("hoi_carry"))
    act = agent_config(settings["bank_root"] + "/rank_0", settings, 20)["actor"]
    act.pop("teleop_checkpoint")
    act["class_name"] = "hoi_carry_actor.ReferenceCarryActor" if original else "hoi_carry_actor.CarryActor"
    if original:
        act["base_checkpoint"] = checkpoint
    ag.update(
        actor=act,
        seed=seed,
        logger="tensorboard",
        upload_model=False,
        save_interval=a.save_interval,
        max_iterations=a.updates,
        amp_dtype=None,
    )
    ag["obs_groups"]["critic"] = ["policy", "augmentation", "hoi"]
    ag["algorithm"].update(
        learning_rate=1e-4,
        schedule="fixed",
        num_learning_epochs=3,
        num_mini_batches=max(2, math.ceil(a.num_envs * ag["num_steps_per_env"] / 1024)),
    )
    trainer = Runner(wrapped, ag, str(a.output), device)
    trainer.alg.broadcast_parameters = MethodType(native_broadcast, trainer.alg)
    actor = trainer.alg._raw_actor
    if a.resume:
        trainer.load(str(a.resume), map_location=device)
        trainer.current_learning_iteration += 1
    elif not original:
        state = load_checkpoint(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
        reduced = {k: v for k, v in state.items() if not k.startswith("teleop_encoder.")}
        missing, extra = actor.load_state_dict(reduced, strict=False)
        if set(missing) != {f"hoi_encoder.{k}" for k in actor.hoi_encoder.state_dict()} or extra:
            raise RuntimeError(f"Incompatible DA2 checkpoint: {missing}, {extra}")
    if a.grasp_feedback:
        actor.hoi_reference_version.fill_(5 if a.visual_pickup else (4 if a.smooth_grasp else 2))
    if a.full_body:
        actor.hoi_reference_version.fill_(3)
    obs = wrapped.get_observations()
    assert obs["hoi"].shape == (a.num_envs, 47) and obs["kv_tokens"]["img_tokens"].shape[1:] == (128, 96)
    assert env.da2_visual.images > 0 and cmd.box.indexing.root_body_id != cmd.robot.indexing.root_body_id
    assert len(cmd.sites) == 2 and not env.scene.env_origins.any()
    info = dict(
        rank=rank,
        world=world,
        envs=a.num_envs,
        seed=seed,
        actor_checkpoint=checkpoint,
        initialization="sonic" if original else "legacy",
        resumed_checkpoint=str(a.resume) if a.resume else None,
        reference_tracking=original,
        grasp_feedback=a.grasp_feedback,
        full_body=a.full_body,
        smooth_grasp=a.smooth_grasp,
        visual_pickup=a.visual_pickup,
        reference_version=int(actor.hoi_reference_version) if original else 0,
        trainable=[k for k, v in actor.named_parameters() if v.requires_grad],
        hoi_dim=obs["hoi"].shape[1],
        visual_shape=list(obs["kv_tokens"]["img_tokens"].shape),
        pid=os.getpid(),
    )
    write_json(a.output / f"init_rank_{rank}.json", info)
    old_log = trainer.logger.log

    def log(**kw):
        old_log(**kw)
        counts = cmd.counts.clone()
        cmd.counts.zero_()
        if world > 1:
            torch.distributed.all_reduce(counts)
        norm = float(actor.hoi_encoder[-1].weight.detach().norm())
        rewards = kw.get("loss_dict", {})
        finite = all(np.isfinite(float(x)) for x in rewards.values() if isinstance(x, (int, float)))
        if not math.isfinite(norm) or not finite:
            raise RuntimeError("Nonfinite PPO update")
        row = dict(
            iteration=kw["it"],
            unix=time.time(),
            seconds=kw["collect_time"] + kw["learn_time"],
            full_ended=int(counts[0]),
            lifted=int(counts[1]),
            carried=int(counts[2]),
            placed=int(counts[3]),
            hoi_projection_norm=norm,
            visual_images=env.da2_visual.images,
        )
        if a.smooth_grasp:
            row.update(nonhand_contact_episodes=int(counts[4]), clean_placed=int(counts[5]))
        if a.visual_pickup:
            far = cmd.far_counts.clone()
            cmd.far_counts.zero_()
            if world > 1:
                torch.distributed.all_reduce(far)
            row.update(
                far_ended=int(far[0]),
                far_lifted=int(far[1]),
                far_placed=int(far[2]),
                far_clean_placed=int(far[3]),
            )
            if rank == 0 and kw["it"] == kw["start_it"]:
                trainer.save(str(a.output / f"model_{kw['it']}.pt"))
        with (a.output / f"progress_rank_{rank}.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    trainer.logger.log = log
    trainer.learn(a.updates - trainer.current_learning_iteration, init_at_random_ep_len=False)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument(
        "--init",
        choices=("sonic", "legacy"),
        help="Defaults to original SONIC for a new run or the saved architecture on resume",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--visual-pickup",
        action="store_true",
        help="Use DA2 pickup localization and retarget the grasp and approach",
    )
    group.add_argument("--grasp-feedback", action="store_true")
    group.add_argument(
        "--smooth-grasp",
        action="store_true",
        help="Two-hand compensation with whole-body tracking, smoothness and non-hand contact costs",
    )
    group.add_argument("--full-body", action="store_true", help="Track all 29 HOI joints including the legs")
    p.add_argument("--resume", type=Path)
    main(p.parse_args())
