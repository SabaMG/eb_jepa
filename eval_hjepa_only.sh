#!/bin/bash
# Quick eval-only of the 2-level H-JEPA on the EXISTING coarse.pth (no retrain), with GIFs
# of the first episodes so we can SEE the failure mode.
#SBATCH --job-name=hjepa_gif
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=hjepa_gif_%j.out
#SBATCH --error=hjepa_gif_%j.err
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312
uv sync --project "$REPO"
ROOT="${EBJEPA_CKPTS:-$EBJEPA_WORK/ckpts}/maze/baseline_full"
ALT="$EBJEPA_WORK/ckpts/maze/baseline_full"
if [ ! -f "$ROOT/aux/latest.pth.tar" ] && [ -f "$ALT/aux/latest.pth.tar" ]; then ROOT="$ALT"; fi
EVAL="$ROOT/eval_hjepa_gif"; mkdir -p "$EVAL"
# <fine> <coarse_pth> <out_dir> num_ep Hc m beam_W budget_factor margin seed n_gifs
uv run --project "$REPO" python -m examples.ac_video_jepa.maze.eval_hjepa \
    "$ROOT/aux/latest.pth.tar" "$ROOT/coarse/coarse.pth" "$EVAL" 200 4 3 4 4 10 0 6
echo "=== GIFs in $EVAL/ep*_*.gif ==="
ls -la "$EVAL"/*.gif 2>/dev/null || echo "(no gifs)"
