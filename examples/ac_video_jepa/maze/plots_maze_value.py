"""Bar chart of the maze planning-cost comparison: learned TD-MPC VALUE vs the
geometric distance costs (probe_pos, repr_dist), grouped by planning regime.
Reads results/maze_value/results.json.

Usage: python -m examples.ac_video_jepa.maze.plots_maze_value <results_dir>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.autolayout": True})
COST_STYLE = {
    "learned_value": ("learned VALUE (TD-MPC, ours)", "#1d6fb8"),
    "probe_pos": ("probe_pos (position distance)", "#e76f51"),
    "repr_dist": ("repr_dist (latent MSE)", "#9aa0a6"),
}


def main():
    rdir = sys.argv[1]
    d = json.load(open(os.path.join(rdir, "results.json")))
    regimes = list(d["regimes"].keys())
    costs = ["learned_value", "probe_pos", "repr_dist"]
    x = np.arange(len(regimes)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, c in enumerate(costs):
        vals = [d["regimes"][r].get(c, np.nan) for r in regimes]
        label, color = COST_STYLE[c]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=color,
                      edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace(" (", "\n(") for r in regimes], fontsize=9)
    ax.set_ylabel("maze success rate (%)  —  16 held-out mazes")
    ax.set_title("Maze planning cost: learned TD-MPC value vs geometric distance\n"
                 "(same frozen world model; only the MPC objective changes)")
    ax.set_ylim(0, 45); ax.legend(loc="upper left")
    out = os.path.join(rdir, "maze_value_compare.png")
    fig.savefig(out); plt.close(fig)
    print(f"[plots_maze_value] wrote {out}")


if __name__ == "__main__":
    main()
