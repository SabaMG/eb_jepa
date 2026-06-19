#!/bin/bash
# Re-run ONLY steps 3 + 4 of the baseline_full pipeline on the existing aux ckpt.
# Use this when base/ and aux/ are already trained and you only need the subgoal
# predictor + the A*-free hierarchical eval (THE REFERENCE NUMBER, ~66% / SPL 0.62).
#
#   3) subgoal     main_subgoal.py   (high level, frozen WM)
#   4) eval        eval_subgoal.py   (A*-free, greedy K-step reacher) -> success% / SPL
#
# Submit:  sbatch rerun_sg_eval.sh
# Read:    the "[subgoal-eval] A*-FREE success=..%  SPL=.." line + $EVAL/subgoal_eval.json
#SBATCH --job-name=maze_sg_eval
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=maze_sg_eval_%j.out
#SBATCH --error=maze_sg_eval_%j.err
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
SG="$ROOT/sg"; EVAL="$ROOT/eval"; FINE="$ROOT/aux/latest.pth.tar"
mkdir -p "$SG" "$EVAL"

if [ ! -f "$FINE" ]; then
    echo "ERROR: aux checkpoint not found at $FINE -- run recreate_baseline_full.sh first." >&2
    exit 1
fi

# --- 3. subgoal predictor (high level, frozen WM) ---------------------------
# Skip if already trained -- lets you re-run the eval alone without redoing ~36 min.
if [ -f "$SG/subgoal.pth.tar" ]; then
    echo ">>> [3/4] subgoal predictor already trained, skipping -> $SG/subgoal.pth.tar"
else
    echo ">>> [3/4] subgoal predictor (N=4, 12 epochs) -> $SG"
    uv run --project "$REPO" python -m examples.ac_video_jepa.maze.main_subgoal \
        "$FINE" "$SG" 4 12
fi

# --- 4. A*-free hierarchical eval = THE REFERENCE NUMBER --------------------
echo ">>> [4/4] A*-free eval (greedy reacher) -> $EVAL"
uv run --project "$REPO" python -m examples.ac_video_jepa.maze.eval_subgoal \
    "$FINE" "$SG/subgoal.pth.tar" "$EVAL" 32 4 0.05 8 4 10

echo "=== DONE. Reference success%/SPL above. subgoal=$SG/subgoal.pth.tar  eval=$EVAL/subgoal_eval.json ==="
