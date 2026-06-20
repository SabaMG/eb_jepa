#!/bin/bash
# DIRECT-METRIC test (Hc=0): bypass the broken macro-option beam, descend the learned
# quasimetric straight toward the goal. Same coarse.pth (the 0.86-monotone 3+4 model),
# seed 0, 200 mazes. Tests whether the good metric ALONE navigates better than the beam.
#SBATCH --job-name=hjepa_direct
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=hjepa_direct_%j.out
#SBATCH --error=hjepa_direct_%j.err
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312
uv sync --project "$REPO"
ROOT="${EBJEPA_CKPTS:-$EBJEPA_WORK/ckpts}/maze/baseline_full"
ALT="$EBJEPA_WORK/ckpts/maze/baseline_full"
if [ ! -f "$ROOT/aux/latest.pth.tar" ] && [ -f "$ALT/aux/latest.pth.tar" ]; then ROOT="$ALT"; fi
EVAL="$ROOT/eval_hjepa_direct"; mkdir -p "$EVAL"
# <fine> <coarse_pth> <out> num_ep Hc=0 m beam_W budget_factor margin seed n_gifs low_depth
uv run --project "$REPO" python -m examples.ac_video_jepa.maze.eval_hjepa \
    "$ROOT/aux/latest.pth.tar" "$ROOT/coarse/coarse.pth" "$EVAL" 200 0 3 4 4 10 0 8 1
