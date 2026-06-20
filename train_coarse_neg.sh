#!/bin/bash
# PARALLEL BET #2: retrain the coarse level WITH cross-trajectory hard negatives (the new
# term in train_coarse.py), then eval it in the same 2-level dream-subgoal mode + low-level
# memory controller. ISOLATED: writes to coarse_neg/ and eval_hjepa_subgoal_neg/ so the
# working coarse.pth (84.5% model) and seed-0 results are untouched.
#SBATCH --job-name=hjepa_neg
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=hjepa_neg_%j.out
#SBATCH --error=hjepa_neg_%j.err
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"; module load python312; uv sync --project "$REPO"
ROOT="${EBJEPA_CKPTS:-$EBJEPA_WORK/ckpts}/maze/baseline_full"
ALT="$EBJEPA_WORK/ckpts/maze/baseline_full"
if [ ! -f "$ROOT/aux/latest.pth.tar" ] && [ -f "$ALT/aux/latest.pth.tar" ]; then ROOT="$ALT"; fi
FINE="$ROOT/aux/latest.pth.tar"
CNEG="$ROOT/coarse_neg"; mkdir -p "$CNEG"            # ISOLATED: does NOT touch the working coarse.pth
EVAL="$ROOT/eval_hjepa_subgoal_neg"; mkdir -p "$EVAL"
echo ">>> [1/2] train coarse WITH cross-trajectory hard negatives -> $CNEG"
uv run --project "$REPO" python -m examples.ac_video_jepa.maze.train_coarse "$FINE" "$CNEG" 20 8
echo ">>> [2/2] A*-free 2-level subgoal eval (memory controller) with the NEG metric -> $EVAL"
uv run --project "$REPO" python -m examples.ac_video_jepa.maze.eval_hjepa \
    "$FINE" "$CNEG/coarse.pth" "$EVAL" 200 4 3 4 4 10 0 8 1 8
echo "=== DONE neg-metric: success/SPL above + $EVAL/hjepa_eval.json ==="
