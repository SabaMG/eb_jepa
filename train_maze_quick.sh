#!/bin/bash
# QUICK maze world model (~20-30 min) for a fast signal to DECIDE the track.
# Small 11x11 mazes + reduced epochs/data. NOT the competition run.
# (Use train_maze_baseline.sh for the real 21x21 run that targets the 66%.)
#
# After this finishes, get a hierarchical number fast (minutes):
#   FINE=$EBJEPA_CKPTS/maze/quick/latest.pth.tar
#   uv run python -m examples.ac_video_jepa.maze.main_subgoal  $FINE $EBJEPA_CKPTS/maze/quick/sg 4 8
#   uv run python -m examples.ac_video_jepa.maze.eval_subgoal  $FINE $EBJEPA_CKPTS/maze/quick/sg/subgoal.pth.tar \
#       $EBJEPA_CKPTS/maze/quick/eval 16 4 0.05 4 4 10
#SBATCH --job-name=maze_quick
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=maze_quick_%j.out
#SBATCH --error=maze_quick_%j.err

set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312
if ! uv --version &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi
uv sync --project "$REPO"

CKPT="${EBJEPA_CKPTS:-$EBJEPA_WORK/ckpts}/maze/quick"
mkdir -p "$CKPT"
echo ">>> QUICK maze WM (11x11, 8 epochs, 40k samples) -> $CKPT"

# Knobs to go faster/slower: --optim.epochs (fewer = faster), --data.size (smaller = faster).
uv run --project "$REPO" python -m examples.ac_video_jepa.main \
    --fname examples/ac_video_jepa/cfgs/train/maze/train_maze_small.yaml \
    --meta.load_model=False \
    --meta.model_folder="$CKPT" \
    --optim.epochs=8 \
    --data.size=40000

echo "=== quick WM done -> $CKPT ==="
