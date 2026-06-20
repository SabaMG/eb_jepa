#!/bin/bash
# CONFIRMATION RUN: same 2-level dream-subgoal config that scored 84.5%/0.325 (job 77313),
# but SEED 1, written to a SEPARATE dir so the seed-0 result/GIFs are NOT overwritten.
#SBATCH --job-name=hjepa_sg1
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=hjepa_sg1_%j.out
#SBATCH --error=hjepa_sg1_%j.err
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"; module load python312; uv sync --project "$REPO"
ROOT="${EBJEPA_CKPTS:-$EBJEPA_WORK/ckpts}/maze/baseline_full"
ALT="$EBJEPA_WORK/ckpts/maze/baseline_full"
if [ ! -f "$ROOT/aux/latest.pth.tar" ] && [ -f "$ALT/aux/latest.pth.tar" ]; then ROOT="$ALT"; fi
EVAL="$ROOT/eval_hjepa_subgoal_s1"; mkdir -p "$EVAL"
# <fine> <coarse> <out> num_ep Hc m beam_W budget_factor margin seed n_gifs low_depth sg_horizon
uv run --project "$REPO" python -m examples.ac_video_jepa.maze.eval_hjepa \
    "$ROOT/aux/latest.pth.tar" "$ROOT/coarse/coarse.pth" "$EVAL" 200 4 3 4 4 10 1 8 1 8
