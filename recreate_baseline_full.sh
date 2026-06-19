#!/bin/bash
# RECREATE THE FULL 21x21 BASELINE (the proven ~66% / SPL 0.62 hierarchical result).
#
# Chained, single job:
#   1) base WM     train_maze_long.yaml  (128-dim, nsteps=32, sample_length=49, from scratch)
#   2) aux-pos FT  train_maze_aux.yaml   (--init_from base; makes the latent position-decodable)
#   3) subgoal     main_subgoal.py       (high level, frozen WM)
#   4) eval        eval_subgoal.py       (A*-free, greedy K-step reacher) -> THE REFERENCE NUMBER
#
# In-training plan evals are OFF (run the eval as step 4); see cluster/DEV_PROCESS.md.
#
# Submit:  sbatch recreate_baseline_full.sh
# Read:    the "[subgoal-eval] A*-FREE success=..%  SPL=.." line at the end + $EVAL/subgoal_eval.json
#SBATCH --job-name=maze_baseline_full
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=maze_baseline_full_%j.out
#SBATCH --error=maze_baseline_full_%j.err

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

ROOT="${EBJEPA_CKPTS:-$EBJEPA_WORK/ckpts}/maze/baseline_full"
BASE="$ROOT/base"; AUX="$ROOT/aux"; SG="$ROOT/sg"; EVAL="$ROOT/eval"
mkdir -p "$BASE" "$AUX" "$SG" "$EVAL"

# --- 1. base world model (long horizon, full 21x21) -------------------------
echo ">>> [1/4] base WM (train_maze_long, plan-eval OFF) -> $BASE"
uv run --project "$REPO" python -m examples.ac_video_jepa.main \
    --fname examples/ac_video_jepa/cfgs/train/maze/train_maze_long.yaml \
    --meta.load_model=False \
    --meta.enable_plan_eval=False \
    --meta.model_folder="$BASE"

# --- 2. aux-position fine-tune (sharper position probe) ----------------------
echo ">>> [2/4] aux-pos fine-tune (init_from base) -> $AUX"
uv run --project "$REPO" python -m examples.ac_video_jepa.main \
    --fname examples/ac_video_jepa/cfgs/train/maze/train_maze_aux.yaml \
    --meta.load_model=False \
    --meta.enable_plan_eval=False \
    --meta.init_from="$BASE/latest.pth.tar" \
    --meta.model_folder="$AUX"

FINE="$AUX/latest.pth.tar"

# --- 3. subgoal predictor (high level, frozen WM) ---------------------------
echo ">>> [3/4] subgoal predictor (N=4, 12 epochs) -> $SG"
uv run --project "$REPO" python -m examples.ac_video_jepa.maze.main_subgoal \
    "$FINE" "$SG" 4 12

# --- 4. A*-free hierarchical eval = THE REFERENCE NUMBER --------------------
echo ">>> [4/4] A*-free eval (greedy reacher) -> $EVAL"
uv run --project "$REPO" python -m examples.ac_video_jepa.maze.eval_subgoal \
    "$FINE" "$SG/subgoal.pth.tar" "$EVAL" 32 4 0.05 8 4 10

echo "=== BASELINE DONE. Reference success%/SPL above. ckpt=$FINE  subgoal=$SG/subgoal.pth.tar ==="
echo "=== Next: sbatch compare_reacher.sh  (greedy vs beam on this same ckpt) ==="
