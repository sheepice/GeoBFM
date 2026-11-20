# Export and sim-to-sim

Run these commands from the repository root after setting up the [training environment and resource profile](../training/README.md). Select the task configuration that produced the checkpoint. The example below uses the raw-SONIC stair recipe and one free GPU.

```bash
STAIRS=configs/tasks/stairs_ab_da2_ours_sonic_raw_20261112.json
python deployment/run.py stairs export --config "$STAIRS" --gpus 0 \
  --checkpoint runs/stairs/model_4000.pt --output exports/stairs
python deployment/run.py stairs verify --config "$STAIRS" --gpus 0 \
  --bundle exports/stairs --output runs/stairs_verify.json
python deployment/run.py stairs sim2sim --config "$STAIRS" --gpus 0 \
  --bundle exports/stairs --seconds 5 --output runs/stairs_sim2sim.json
```

The export includes the RGB encoder, ONNX policy, camera settings and parity inputs. Verification exercises the encoder and the exported RGB-to-action path. Sim-to-sim reads the camera contract from the bundle and runs the policy in MuJoCo. For stepping stones, select the `stones` task and its matching configuration, checkpoint and output paths. [Export code](../training/src/vibe_grail_joint/da2_locomotion/export.py) · [Camera contract](../configs/camera_d435i_112_fov42p5.json)
