#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
# Choose the Torch CUDA build for the host before invoking this installer if needed.
.venv/bin/python -m pip install -r requirements.txt
if [ ! -d external/sim-assets/.git ]; then
  mkdir -p external
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/drcl-deploy/assets.git external/sim-assets
fi
git -C external/sim-assets checkout --detach 90e226008bda5edc9c07ca3eb69f9f79c4603de1
# Robot meshes are external data, never added to this repository.
git -C external/sim-assets lfs pull --include='src/assets/g1/**'
.venv/bin/python -m pip install -e external/sim-assets
# mjlab requests upstream rsl-rl 5.4.0. Install the required SONIC fork last,
# without dependency co-resolution, as in the validated source experiment.
.venv/bin/python -m pip install --no-deps -e third_party/rsl_rl -e third_party/mocke -e third_party/orcs -e third_party/vibe -e .
if [ ! -e configs/paths.local.json ]; then
  cp configs/paths.example.json configs/paths.local.json
fi
echo 'Environment ready. Set external resources in configs/paths.local.json, then run scripts/run.py stairs doctor.'
