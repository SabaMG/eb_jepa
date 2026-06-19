#!/bin/bash
# COMPARE LOW-LEVEL REACHERS on the recreated baseline (eval-only, frozen WM).
# flat  vs  hier_greedy (baseline L1)  vs  hier_beam (the improvement),
# same WM + subgoal + same seeded mazes -> the success/SPL deltas isolate the reacher.
#
# Needs recreate_baseline_full.sh to have finished (uses its aux + subgoal ckpts).
# Submit:  sbatch compare_reacher.sh
# Read:    the "REACHER COMPARISON" table at the end + $EVAL/beam_compare.json
#SBATCH --job-name=maze_reacher_cmp
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=maze_reacher_cmp_%j.out
#SBATCH --error=maze_reacher_cmp_%j.err

set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"

echo "=== Host: $(hostname) | Arch: $(uname -m) | Date: $(date) ==="
module load python312
if ! uv --version &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi
uv sync --project "$REPO"

ROOT="${EBJEPA_CKPTS:-$EBJEPA_WORK/ckpts}/maze/baseline_full"
FINE="$ROOT/aux/latest.pth.tar"
SG="$ROOT/sg/subgoal.pth.tar"
EVAL="$ROOT/eval_reacher_cmp"
mkdir -p "$EVAL"

if [ ! -f "$FINE" ] || [ ! -f "$SG" ]; then
    echo "ERROR: missing $FINE or $SG. Run recreate_baseline_full.sh first." >&2
    exit 1
fi

# args: num_ep lookahead beam_depth beam_width revisit_pen n_gifs budget_factor margin seed
echo ">>> reacher comparison (flat / hier_greedy / hier_beam) -> $EVAL"
uv run --project "$REPO" python -m examples.ac_video_jepa.maze.eval_beam \
    "$FINE" "$SG" "$EVAL" 32 4 6 8 0.05 6 4 10 0

echo "=== reacher comparison done. Read the table above + $EVAL/beam_compare.json ==="
