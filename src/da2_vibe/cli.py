"""One guarded launcher for stairs, round stepping stones and optional TELEOP."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from .paths import load_settings, profile, profile_path, repo_root, resolve

ENTRYPOINTS = {
    ("stairs", "train"): "vibe_grail_joint.da2_flat_repair.train",
    ("stairs", "evaluate"): "vibe_grail_joint.da2_locomotion.evaluate",
    ("stairs", "audit"): "vibe_grail_joint.da2_locomotion.stair_failure_audit",
    ("stairs", "visual-audit"): "vibe_grail_joint.da2_locomotion.groot_audit",
    ("stones", "visual-audit"): "vibe_grail_joint.da2_locomotion.groot_audit",
    ("stones", "train"): "vibe_grail_joint.da2_stones.train",
    ("stones", "evaluate"): "vibe_grail_joint.da2_stones.evaluate",
    ("stones", "review"): "vibe_grail_joint.da2_stones.review",
    ("stones", "stress"): "vibe_grail_joint.da2_stones.stress",
    ("teleop", "train"): "vr_vibe.da2_train",
}
for _task in ("stairs", "stones"):
    for _action in ("export", "verify", "sim2sim"):
        ENTRYPOINTS[_task, _action] = "vibe_grail_joint.da2_locomotion." + _action


def doctor(task, settings):
    resources = [("planner", settings["planner"])]
    if settings['visual']['backend'] in ('da2_metric_small', 'da2_metric_base', 'da2_metric_large'):
        resources += [("DA2 source", str(Path(settings["visual"]["source"]) / "metric_depth/depth_anything_v2/dinov2.py")),
                      ("DA2 weights", settings["visual"]["checkpoint"])]
    for key in ("stage1_checkpoint", "sonic_checkpoint", "initial_checkpoint"):
        if key in settings:
            resources.append((key, settings[key]))
    if settings.get("dagger"):
        resources.append(("SONIC teacher", settings["dagger"]["teacher_checkpoint"]))
    if settings.get('flat_training'):
        resources.append(('flat stand/start-stop references', settings['flat_training']['reference_bank']))
    if settings.get('groot_visual'):
        resources.append(('HDR environment bank', settings['groot_visual']['bank']))
    if task == "teleop":
        resources.append(("TELEOP encoder", settings["vr_teleop"]["encoder_checkpoint"]))
    if "bank_root" in settings:
        for rank in range(settings["expected_world_size"]):
            resources.append((f"bank rank {rank}", str(Path(settings["bank_root"]) / f"rank_{rank}/bank.json")))
    else:
        resources.append(("schema bank", str(Path(settings["schema_bank"]) / "bank.json")))
    rows = [dict(resource=name, path=value, exists=Path(value).is_file()) for name, value in resources]
    print(json.dumps(dict(task=task, profile=str(profile_path()), resources=rows,
                         ok=all(r["exists"] for r in rows)), ensure_ascii=False, indent=2))
    return 0 if all(r["exists"] for r in rows) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("stairs", "stones", "teleop"))
    parser.add_argument("action", choices=("train", "evaluate", "audit", "visual-audit", "review", "stress", "export", "verify", "sim2sim", "doctor"))
    parser.add_argument("--paths", type=Path, default=profile_path())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--gpus", help="Physical CUDA indices, e.g. 0,1,2,3")
    parser.add_argument("--dry-run", action="store_true", help="Print command and resolved config; do not use GPUs")
    args, rest = parser.parse_known_args(argv)
    if rest[:1] == ["--"]:
        rest = rest[1:]
    os.environ["DA2_VIBE_PATHS"] = str(args.paths.resolve())
    config = args.config or repo_root() / f"configs/tasks/{args.task}.json"
    settings = load_settings(config)
    if args.action == "doctor":
        return doctor(args.task, settings)
    module = ENTRYPOINTS.get((args.task, args.action))
    if args.task == "stones" and args.action == "evaluate" and "--paired" in rest:
        module = "vibe_grail_joint.da2_stones.paired_evaluation"
        rest.remove("--paired")
    if args.task == 'stones' and args.action == 'sim2sim' and '--video' in rest:
        module = 'vibe_grail_joint.da2_stones.sim2sim_video'
        rest.remove('--video')
    if module is None:
        parser.error(f"{args.task} has no validated {args.action} entry point")
    if not args.gpus:
        parser.error("--gpus is required")
    indices = [int(x) for x in args.gpus.split(",")]
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0:
        parser.error("Select distinct nonnegative GPU indices")
    if args.action == "train":
        from .ablations import single_gpu_ablation
        single_ablation = args.task == 'stairs' and single_gpu_ablation(settings)
        if len(indices) not in (2, 4, 8) and not (len(indices) == 1 and single_ablation):
            parser.error("Training requires 2, 4 or 8 ranks, or an explicit single-GPU stairs ablation")
        if "--smoke" not in rest and len(indices) != settings["expected_world_size"]:
            parser.error("GPU count must match expected_world_size and reference bank sharding")
        if "--log-dir" not in rest:
            parser.error("Training requires an explicit new --log-dir, or --resume into an existing run")
    elif len(indices) != 1:
        parser.error("Evaluation/export uses one GPU")
    command = [sys.executable]
    if args.action == "train":
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(indices)}"]
    command += [str(repo_root() / "scripts/worker.py"), module]
    # Only sim2sim takes its settings from the exported bundle.
    needs_config = args.action != "sim2sim"
    if needs_config:
        command += ["--config", str(config.resolve())]
    command += [resolve(x) for x in rest]
    if args.dry_run:
        print(json.dumps(dict(command=command, settings=settings, gpus=indices,
                              gpu_lock_dir=resolve(profile().get("gpu_lock_dir", "${CACHE_ROOT}/gpu-locks"))),
                         ensure_ascii=False, indent=2))
        return 0
    from vibe_grail_joint.gpu_guard import assert_gpus_idle
    lock_dir = Path(resolve(profile().get("gpu_lock_dir", "${CACHE_ROOT}/gpu-locks")))
    lock_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(resolve("${CACHE_ROOT}"))
    cache.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for index in sorted(indices):
            handle = (lock_dir / f"gpu_{index}.lock").open("a")
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_gpus_idle(indices)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   VIBE_JOINT_GUARD_PID=str(os.getpid()), OMP_NUM_THREADS="2",
                   PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                   HF_HOME=str(cache / "hf"), HF_HUB_OFFLINE="1",
                   TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), WARP_CACHE_PATH=str(cache / "warp"),
                   MPLCONFIGDIR=str(cache / "matplotlib"), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        if args.action == "train":
            output = Path(command[command.index("--log-dir") + 1]).resolve()
            output.mkdir(parents=True, exist_ok=True)
            target = output / "resolved_config.json"
            if target.exists() and json.loads(target.read_text()) != settings:
                raise ValueError("Existing run has a different resolved config; use a new run directory")
            target.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
        child = subprocess.Popen(command, env=env)
        def forward(signum, _frame):
            if child.poll() is None:
                child.send_signal(signum)
        signal.signal(signal.SIGINT, forward)
        signal.signal(signal.SIGTERM, forward)
        return child.wait()
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
