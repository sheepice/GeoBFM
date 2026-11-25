"""Match the three sampling budgets in the ancestry of the DA2 HOI model_2500."""

import argparse
import json
import math
import os
import time
from pathlib import Path
from types import MethodType

import _bootstrap
import torch
from da2_vibe.paths import load_settings
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from hoi_training import Runner, native_broadcast, write_json
from hoi_carry_task import config
from hoi_native_vibe import native_agent, native_observations, segmented_returns

# Inclusive checkpoint interval, actors per GPU, 24-step chunks per update, minibatches per epoch.
STAGES = {
    "initial": (0, 600, 96, 1, 9),
    "grasp": (601, 2000, 9472, 1, 444),
    "pickup": (2001, 2500, 2240, 16, 420),
}


def main(a):
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 4
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    device = f"cuda:{rank}"
    seed = 20261124 + rank
    torch.manual_seed(seed)
    start, end, n, chunks, minibatches = STAGES[a.stage]
    pickup = a.stage == "pickup"
    settings = load_settings(str(_bootstrap.ROOT / "configs/tasks/teleop.json"))
    ckpt = settings["dagger"]["teacher_checkpoint"]
    cfg = config(
        a.bundle,
        settings,
        n,
        seed,
        reference_tracking=True,
        grasp_feedback=True,
        smooth_grasp=pickup,
        visual_pickup=pickup,
    )
    raw = ManagerBasedRlEnv(native_observations(cfg), device)
    env = RslRlVecEnvWrapper(raw, clip_actions=None)
    ag = native_agent(ckpt)
    ag.update(
        num_steps_per_env=24 * chunks,
        seed=seed,
        logger="tensorboard",
        upload_model=False,
        save_interval=50,
        max_iterations=end + 1,
        amp_dtype=None,
    )
    ag["algorithm"].update(
        learning_rate=1e-4, schedule="fixed", num_learning_epochs=3, num_mini_batches=minibatches
    )
    a.output.mkdir(exist_ok=True, parents=True)
    if rank == 0:
        # The registry follows the current stage across fresh torchrun processes.
        pointer = a.output.parent / "current_progress.jsonl"
        pending = pointer.with_suffix(".new")
        pending.unlink(missing_ok=True)
        pending.symlink_to(Path(a.output.name) / "progress.jsonl")
        pending.replace(pointer)
    trainer = Runner(env, ag, str(a.output), device)
    alg = trainer.alg
    alg.broadcast_parameters = MethodType(native_broadcast, alg)
    if chunks > 1:
        alg.compute_returns = MethodType(segmented_returns, alg)
    actor = alg._raw_actor
    if a.resume:
        trainer.load(str(a.resume), map_location=device)
        trainer.current_learning_iteration += 1
    if not start <= trainer.current_learning_iteration <= end:
        raise ValueError(f"Incompatible resume iteration for {a.stage}")
    actor.hoi_reference_version.fill_(5 if pickup else 2)
    obs = env.get_observations()
    assert obs["kv_tokens"]["img_tokens"].shape[1:] == (49, 192) and obs["hoi"].shape[-1] == 47
    assert raw.da2_visual.images > 0
    raw.da2_visual.encoder.assert_frozen()
    write_json(
        a.output / f"init_rank_{rank}.json",
        dict(
            rank=rank,
            world=world,
            stage=a.stage,
            seed=seed,
            pid=os.getpid(),
            envs=n,
            segments=chunks,
            global_steps_per_update=world * n * 24 * chunks,
            actor_checkpoint=ckpt,
            resume=str(a.resume) if a.resume else None,
            trainable=[k for k, v in actor.named_parameters() if v.requires_grad],
            minibatches=minibatches,
            actor=ag["actor"],
            algorithm=ag["algorithm"],
        ),
    )
    old_log = trainer.logger.log
    cmd = raw.command_manager.get_term("motion")

    def log(**kw):
        old_log(**kw)
        if not all(math.isfinite(float(v)) for v in kw["loss_dict"].values()):
            raise RuntimeError("Nonfinite PPO loss")
        counts = cmd.counts.clone()
        cmd.counts.zero_()
        torch.distributed.all_reduce(counts)
        if rank == 0:
            row = dict(
                iteration=kw["it"],
                stage=a.stage,
                global_steps_per_update=world * n * 24 * chunks,
                seconds=kw["collect_time"] + kw["learn_time"],
                unix=time.time(),
                full_ended=int(counts[0]),
                lifted=int(counts[1]),
                carried=int(counts[2]),
                placed=int(counts[3]),
                loss=kw["loss_dict"],
            )
            if pickup:
                row.update(clean_placed=int(counts[5]))
            with (a.output / "progress.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")

    trainer.logger.log = log
    trainer.learn(end + 1 - trainer.current_learning_iteration)
    torch.distributed.destroy_process_group()
    raw.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    main(p.parse_args())
