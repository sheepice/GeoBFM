# GeoBFM

**Visual Geometry-Grounded Adaptation for Perceptive Behavior Foundation Model**

GeoBFM adapts a pretrained SONIC whole-body motion tracker for the Unitree G1. A head-mounted RGB camera observes the next tread or foothold while a planner supplies a nominal motion reference. This repository contains the training environments for stairs and cylindrical stepping stones, the visual policy, evaluation code, and a MuJoCo workflow for running exported policies.

## How the policy works

A frozen Depth Anything V2 Small encoder extracts patch features from a 112 x 112 RGB view with a 42.5-degree horizontal and vertical field of view. Two frames provide 128 patch tokens. Proprioception and the velocity command each form an attention query. The resulting features condition low-rank adapters in the frozen tracker, which outputs 29 joint actions. During PPO training, a 99-point terrain-height prediction objective supervises the visual representation. At inference, the policy takes RGB, robot state and command inputs.

## Start here

| Goal | Entry point |
| --- | --- |
| Install dependencies, configure external assets and train | [Training guide](training/README.md) |
| Train stairs from raw SONIC with flat-ground planner references | [Stairs recipe](configs/tasks/stairs_ab_da2_ours_sonic_raw_20261112.json) |
| Train ordered, alternating stepping-stone traversal from raw SONIC | [Stepping-stone recipe](configs/tasks/stones_ab_da2_ours_sonic_raw_20261117.json) |
| Evaluate stair contact and heading quality | [Scoring code](scripts/report_stairs_quality.py) and [thresholds](configs/evaluation/stairs_quality_v1.json) |
| Export a trained policy, check it and run it in simulation | [Deployment guide](deployment/README.md) |
| Explore the upper-body extension | [Three-point teleoperation](training/src/vr_vibe) and [pick-and-carry configuration](configs/demos/revo2_pick_carry.json) |

The two recipes above train separate policies. Both start with raw SONIC and fresh visual adaptation parameters. Historical recipes that start from the ViBe stage-one checkpoint, matched ViBe baselines and architecture ablations are also available in [configs/tasks](configs/tasks).

## Repository layout

```text
training/       Policy, PPO, environments and evaluation
deployment/     Export, verification and sim-to-sim entry point
src/da2_vibe/   Shared launcher and resource configuration
configs/        Task recipes, camera calibration and evaluation settings
scripts/        Environment setup, reference-bank preparation and reporting tools
tests/          Tests for paths, visual fusion and stair scoring
third_party/    Pinned dependency source, provenance and upstream licenses
```

The Python module names used by the original experiments are preserved for checkpoint compatibility. The training and deployment layout follows [GeoLoco](https://github.com/liuyufei-nubot/GeoLoco).

## Resources and licensing

Checkpoints, motion data, robot assets and rollout videos are external resources. The training guide lists the files needed for each recipe. The original project code retains its [existing license status](LICENSE.md). Bundled third-party components retain their [upstream licenses and notices](THIRD_PARTY_NOTICES.md).
