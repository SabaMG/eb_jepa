#!/bin/bash
# SEED-1 confirmation of the EVERY-STEP dream-subgoal controller (current HEAD, seed-0 = 85.5%/0.341).
# Separate dir so neither the seed-0 every-step nor the seed-1 baseline (85.0%) result is overwritten.
#SBATCH --job-name=hjepa_v2s1
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=hjepa_v2s1_%j.out
#SBATCH --error=hjepa_v2s1_%j.err
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"; module load python312; uv sync --project "$REPO"
ROOT="${EBJEPA_CKPTS:-$EBJEPA_WORK/ckpts}/maze/baseline_full"
ALT="$EBJEPA_WORK/ckpts/maze/baseline_full"
if [ ! -f "$ROOT/aux/latest.pth.tar" ] && [ -f "$ALT/aux/latest.pth.tar" ]; then ROOT="$ALT"; fi
EVAL="$ROOT/eval_hjepa_subgoal_v2_s1"; mkdir -p "$EVAL"
# <fine> <coarse> <out> num_ep Hc m beam_W budget_factor margin seed n_gifs low_depth sg_horizon
uv run --project "$REPO" python -m examples.ac_video_jepa.maze.eval_hjepa \
    "$ROOT/aux/latest.pth.tar" "$ROOT/coarse/coarse.pth" "$EVAL" 200 4 3 4 4 10 1 8 1 8
