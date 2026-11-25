"""Fit a first-person pickup localizer using rendered, independently shifted box poses."""

import argparse
import json
from pathlib import Path

import _bootstrap
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from da2_vibe.paths import load_settings
from hoi_carry_task import config
from hoi_pickup_vision import ground_intersection, image_features, predict_ray


@torch.no_grad()
def main(args):
    seed = 20261125
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    settings = load_settings(str(_bootstrap.ROOT / "configs/tasks/teleop.json"))
    raw = ManagerBasedRlEnv(config(args.bundle, settings, 256, seed, evaluate=True), "cuda:0")
    try:
        raw.reset()
        c = raw.command_manager.get_term("motion")
        sensor = raw.scene["head_cam"]
        cam = sensor.camera_idx
        with np.load(args.bundle / "motion.npz") as f:
            poses = torch.as_tensor(f["qpos"][:16], device=raw.device, dtype=torch.float32)
        batches = []
        for _ in range(88):
            robot = poses[torch.randint(len(poses), (256,), device=raw.device)].clone()
            robot[:, :3] += (torch.rand_like(robot[:, :3]) - 0.5) * 0.055
            robot[:, 7:] += (torch.rand_like(robot[:, 7:]) - 0.5) * 0.10
            box = c.box_pose[:1].expand(256, -1).clone()
            displacement = torch.rand_like(box[:, :2]) * box.new_tensor([0.50, 0.46]) - box.new_tensor(
                [0.16, 0.23]
            )
            box[:, :2] += displacement @ c.box_axes
            raw.sim.data.qpos[:].copy_(torch.cat([robot, box], -1))
            raw.sim.forward()
            raw.sim.sense()
            tokens = raw.da2_visual.encoder(sensor.data.rgb)
            origin = raw.sim.data.cam_xpos[:, cam].clone()
            rotation = raw.sim.data.cam_xmat[:, cam].reshape(-1, 3, 3).clone()
            cam_box = ((box[:, None, :3] - origin[:, None, :]) @ rotation).squeeze(1)
            rays = cam_box[:, :2] / -cam_box[:, 2:]
            batches.append((image_features(tokens), rays, origin, rotation, box[:, :3]))
        features, rays, origins, rotations, boxes = map(torch.cat, zip(*batches, strict=True))
        split = 80 * 256
        x, y = features[:split], rays[:split]
        mean, std = x.mean(0), x.std(0).clamp_min(0.02)
        x = (x - mean) / std
        head = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 192), torch.nn.SiLU(), torch.nn.Linear(192, 2)
        ).to(raw.device)
        opt = torch.optim.AdamW(head.parameters(), lr=0.0006)
        with torch.enable_grad():
            for epoch in range(75):
                if epoch in (40, 60):
                    opt.param_groups[0]["lr"] *= 0.35
                for ids in torch.randperm(len(x), device=raw.device).split(256):
                    loss = (head(x[ids]) - y[ids] * 5).square().mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        calibration = dict(
            mean=mean,
            std=std,
            hidden=head[0].weight,
            hidden_bias=head[0].bias,
            output=head[2].weight,
            output_bias=head[2].bias,
        )
        pred = predict_ray(features[split:], calibration)
        xy = ground_intersection(pred, rotations[split:], origins[split:], boxes[split:, 2])
        error = (xy - boxes[split:, :2]).norm(dim=1)
        scores = dict(
            seed=seed,
            fit=split,
            heldout=len(error),
            mean_cm=100 * error.mean().item(),
            p95_cm=100 * error.quantile(0.95).item(),
            max_cm=100 * error.max().item(),
            encoder_sha256=settings["visual"]["checkpoint_sha256"],
        )
        print(scores, flush=True)
        args.report.write_text(json.dumps(scores, indent=2) + "\n")
        if not torch.isfinite(error).all() or scores["p95_cm"] > 2.0:
            raise RuntimeError("Pickup localization missed the 2 cm held-out tolerance")
        torch.save({k: v.detach().cpu() for k, v in calibration.items()}, args.bundle / "pickup_vision.pt")
    finally:
        raw.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    main(p.parse_args())
