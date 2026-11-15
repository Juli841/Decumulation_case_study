"""Frontier of strategies/run.py (trained once per alpha) and bengen_optimised (also trained per alpha) against fixed rules: Bengen 4%, funded ratio, guardrails.

    python my_frontier.py

Saves frontier_mine.png and prints test ruin_prob and norm_wd. The fixed rules are single points (nothing to tune).
"""

import contextlib
import dataclasses
import io
import os

import matplotlib.pyplot as plt

from main import BASELINE, FRONTIER_ALPHAS, HERE, fit_and_score
from utils import Config, load_scenarios

S = lambda name: os.path.join(HERE, "strategies", name)
RULES = {"Bengen 4%": BASELINE, "funded ratio": S("funded_ratio.py"), "guardrails": S("guardrails.py")}


def quiet_score(path, train, test, cfg):
    with contextlib.redirect_stdout(io.StringIO()):  # hide the training output
        return fit_and_score(path, train, test, cfg)["test"]


def main():
    cfg = Config()
    train = load_scenarios(os.path.join(HERE, "data", "train.parquet"))
    test = load_scenarios(os.path.join(HERE, "data", "validation.parquet"))

    def curve(name, path):
        pts = []
        for alpha in FRONTIER_ALPHAS:
            r = quiet_score(path, train, test, dataclasses.replace(cfg, alpha=alpha))
            print(f"  {name} alpha {alpha}: ruin_prob {r['ruin_prob']:.4f}, norm_wd {r['norm_wd']:.4f}")
            pts.append((r["ruin_prob"], r["norm_wd"], alpha))
        return sorted(pts)

    mine = curve("run.py", S("run.py"))
    opt = curve("bengen_optimised", S("bengen_optimised.py"))
    rules = {name: quiet_score(path, train, test, cfg) for name, path in RULES.items()}
    for name, r in rules.items():
        print(f"  {name}: ruin_prob {r['ruin_prob']:.4f}, norm_wd {r['norm_wd']:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot([p[0] for p in mine], [p[1] for p in mine], "-s", label="run.py (trained per alpha)")
    ax.plot([p[0] for p in opt], [p[1] for p in opt], "-o", label="bengen_optimised (trained per alpha)")
    for pts in (mine, opt):
        for x, y, alpha in pts:
            ax.annotate(f"{alpha}", (x, y), xytext=(5, -9), textcoords="offset points", fontsize=7)
    for (name, r), marker in zip(rules.items(), "*^v"):
        ax.scatter(r["ruin_prob"], r["norm_wd"], marker=marker, s=150, zorder=3, label=name)
    ax.set_xlabel("ruin probability (test)")
    ax.set_ylabel("normalised withdrawals, norm_wd (test)")
    ax.set_title("run.py vs rule-based strategies (better: up and left)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(os.path.join(HERE, "frontier_mine.png"), dpi=150, bbox_inches="tight")
    print("\nSaved the plot to frontier_mine.png")


if __name__ == "__main__":
    main()
