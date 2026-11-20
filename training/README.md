# Training and evaluation

Run the commands below from the repository root. The simulator uses Linux, NVIDIA CUDA, Python 3.11, mjlab and MuJoCo. Install Git LFS for the G1 meshes and choose a CUDA-enabled PyTorch build compatible with the host. The [tested versions](../requirements-tested.txt) record the research environment.

## Preview a run

These commands use the Python standard library and print the resolved settings without starting a worker or opening a GPU.

```bash
STAIRS=configs/tasks/stairs_ab_da2_ours_sonic_raw_20261112.json
STONES=configs/tasks/stones_ab_da2_ours_sonic_raw_20261117.json
python3 scripts/run.py stairs train --paths configs/paths.example.json --config "$STAIRS" \
  --gpus 0,1,2,3 --log-dir runs/stairs --dry-run
python3 scripts/run.py stones train --paths configs/paths.example.json --config "$STONES" \
  --gpus 0,1,2,3 --log-dir runs/stones --dry-run
```

## Install and locate assets

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

The installer prepares an editable environment, fetches the pinned G1 meshes and creates an ignored `configs/paths.local.json` from the [example profile](../configs/paths.example.json). Edit the local profile to point to the following assets. Paths under `${ASSET_ROOT}` default to the checkout's `external/` directory. Set `DA2_VIBE_ASSET_ROOT` to use another location.

| Recipe | Required resources in the local profile |
| --- | --- |
| Both primary recipes | `sonic_checkpoint` (ported SONIC tracker), `planner` (SONIC ONNX planner), `da2_checkpoint` (metric Hypersim Small weights) and the bundled `da2_source` |
| Stairs | `bank_4gpu` containing four compatible reference-bank shards, each with its own `rank_N/bank.json` and referenced arrays |
| Stepping stones | `schema_bank` pointing to one compatible shard directory containing `bank.json` and its arrays |
| Historical stage-one recipes | `stage1_checkpoint` in addition to the resources referenced by the selected JSON |

The exact SHA-256 for the visual weights is recorded in each primary recipe. The reference-bank builder accepts external motion and terrain inputs listed in [configs/data.json](../configs/data.json). The model weights and prepared experiment banks must be supplied separately. [Depth Anything V2 Metric Hypersim Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small)

## Train

Using the recipe variables above, check the assets before starting a run. Select four free GPUs matching the bank shards. The environment counts in the recipes are per GPU and reflect the original experiment's memory budget.

```bash
python scripts/run.py stairs doctor --config "$STAIRS"
python scripts/run.py stones doctor --config "$STONES"

# Run each command on its own set of free GPUs, or run them sequentially.
python scripts/run.py stairs train --config "$STAIRS" --gpus 0,1,2,3 --log-dir runs/stairs
python scripts/run.py stones train --config "$STONES" --gpus 0,1,2,3 --log-dir runs/stones
```

Both recipes use flat-ground planner references on the target terrain and train through the zero-indexed checkpoint `model_4000.pt`. Select the recipe explicitly. The default `stairs.json` and `stones.json` describe earlier stage-one-initialized experiments.

## Evaluate

This example runs regular stair ascent at four riser heights and records contact and heading traces. The route, checkpoint, heights and command speeds are explicit so that they can be shared across policies.

```bash
python scripts/run.py stairs evaluate --config "$STAIRS" --gpus 0 \
  --checkpoint runs/stairs/model_4000.pt --output runs/eval/stairs_up \
  --route up --heights .12,.16,.20,.25 --speeds .25,.4,.7 \
  --measure-foot-contact --strict-quality-trace
```

The [stair evaluator](src/vibe_grail_joint/da2_locomotion/evaluate.py) also supports descent and irregular stairs. The [paired stepping-stone evaluator](src/vibe_grail_joint/da2_stones/paired_evaluation.py) runs through `python scripts/run.py stones evaluate --paired`. It records ordered landings, support and completion on the same course definitions across methods.
