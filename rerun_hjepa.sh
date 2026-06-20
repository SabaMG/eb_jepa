#!/bin/bash
# Strict 2-level H-JEPA: train the LEVEL-1 coarse JEPA (psi + P_high) on the frozen
# fine WM via WM dreams, then run the A*-free 2-level eval. WM is NOT retrained.
#SBATCH --job-name=maze_hjepa
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=maze_hjepa_%j.out
#SBATCH --error=maze_hjepa_%j.err
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312
uv sync --project "$REPO"

ROOT="${EBJEPA_CKPTS:-$EBJEPA_WORK/ckpts}/maze/baseline_full"
ALT="$EBJEPA_WORK/ckpts/maze/baseline_full"
if [ ! -f "$ROOT/aux/latest.pth.tar" ] && [ -f "$ALT/aux/latest.pth.tar" ]; then ROOT="$ALT"; fi
FINE="$ROOT/aux/latest.pth.tar"
COARSE="$ROOT/coarse"; mkdir -p "$COARSE"
EVAL="$ROOT/eval_hjepa"; mkdir -p "$EVAL"
if [ ! -f "$FINE" ]; then echo "ERROR: aux ckpt not found at $FINE" >&2; exit 1; fi

echo ">>> [1/2] train level-1 coarse JEPA (dreams) -> $COARSE"
# <fine_ckpt> <out_dir> <epochs> <k macro-horizon>
uv run --project "$REPO" python -m examples.ac_video_jepa.maze.train_coarse "$FINE" "$COARSE" 10 5

echo ">>> [2/2] A*-free 2-level eval -> $EVAL"
# <fine> <coarse_pth> <out_dir> num_ep Hc m beam_W budget_factor margin seed
uv run --project "$REPO" python -m examples.ac_video_jepa.maze.eval_hjepa \
    "$FINE" "$COARSE/coarse.pth" "$EVAL" 200 4 3 4 4 10 0

echo "=== DONE. 2-level success%/SPL above + $EVAL/hjepa_eval.json ==="
