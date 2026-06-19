#!/bin/bash
# Train the BASELINE maze fine world model (the base everything builds on:
# flat planning AND the hierarchical subgoal). From scratch with the proven long
# config (min_path 50, sample_length 49, 12 epochs).
#
# Submit:  sbatch train_maze_baseline.sh
# Output:  $EBJEPA_CKPTS/maze/baseline/{latest.pth.tar, config.yaml, ...}
#
# Proven 2-stage path for the 66% hierarchical result:
#   1) this script (long base WM)
#   2) aux-pos fine-tune:  --fname .../train_maze_aux.yaml --meta.init_from=<this ckpt>
#      (adds aux_pos -> sharper position probe -> better subgoal/reacher)
#SBATCH --job-name=maze_baseline
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=maze_baseline_%j.out
#SBATCH --error=maze_baseline_%j.err

set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"

echo "=== Host: $(hostname) | Arch: $(uname -m) | Date: $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

module load python312
if ! uv --version &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi
echo ">>> uv sync..."
uv sync --project "$REPO"

CONFIG="examples/ac_video_jepa/cfgs/train/maze/train_maze_long.yaml"
CKPT="${EBJEPA_CKPTS:-$EBJEPA_WORK/ckpts}/maze/baseline"
mkdir -p "$CKPT"
echo ">>> training baseline maze WM -> $CKPT"

uv run --project "$REPO" python -m examples.ac_video_jepa.main \
    --fname "$CONFIG" \
    --meta.load_model=False \
    --meta.model_folder="$CKPT"

echo "=== baseline maze WM done -> $CKPT ==="
