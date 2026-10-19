#!/usr/bin/env python3
"""Regenerates figures/figure2_open_data_scores.png from outputs/norm_specific_open_data_scores_wide.csv.
Requires matplotlib. Run after code/run_replication.py."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(ROOT / "outputs" / "norm_specific_open_data_scores_wide.csv")))
q = [int(r["q"]) for r in rows]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for key, lab, marker, ls in [("p1", "$p=1$", "o", "-"), ("p2", "$p=2$", "s", "--"), ("pinf", r"$p=\infty$", "^", ":")]:
    ax.plot(q, [float(r[key]) for r in rows], marker=marker, ls=ls, lw=1.8, label=lab)
ax.axhline(0.55, color="k", lw=0.9, ls="-.", label="$c_0=0.55$")
ax.set_xlabel("Ordered geometry profile $q$")
ax.set_ylabel("TOPSIS closeness score")
ax.set_xticks(q)
ax.grid(ls=":", alpha=0.5)
ax.legend(frameon=False, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "figure2_open_data_scores.png", dpi=300)
