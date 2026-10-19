#!/usr/bin/env python3
"""Figure 4 (simulation study) from outputs/simulation_study_cells.csv and simulation_study_spearman_bins.csv.
Requires matplotlib. Run after code/simulation_study.py."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
cells = list(csv.DictReader(open(ROOT / "outputs" / "simulation_study_cells.csv")))
bins = list(csv.DictReader(open(ROOT / "outputs" / "simulation_study_spearman_bins.csv")))
fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.6))
for n, mk, ls in [(6, "o", "-"), (12, "s", "--"), (24, "^", ":")]:
    rows = [r for r in cells if r["norm"] == "p2" and r["m_criteria"] == "6" and r["pi_conflict"] == "0.0" and r["n_alternatives"] == str(n)]
    a.plot([float(r["delta"]) for r in rows], [float(r["mean_invalid_band_share"]) for r in rows], marker=mk, ls=ls, lw=1.8, label=f"$N+1={n}$")
a.set_xlabel(r"Trend per step $\delta$ (noise s.d. = 1)"); a.set_ylabel("Invalid share of score range")
a.set_title("(a) $p=2$, $m=6$, no conflicting criteria", fontsize=10); a.legend(frameon=False); a.grid(ls=":", alpha=.5)
labels = ["<0.5", "0.5-0.8", "0.8-0.9", "0.9-0.95", ">=0.95"]
for k, lab, mk, ls in [("p1", "$p=1$", "o", "-"), ("p2", "$p=2$", "s", "--"), ("pinf", r"$p=\infty$", "^", ":")]:
    rows = [r for r in bins if r["norm"] == k]
    b.plot(range(len(rows)), [float(r["P_median_cutoff_invalid"]) for r in rows], marker=mk, ls=ls, lw=1.8, label=lab)
b.set_xticks(range(5)); b.set_xticklabels(labels); b.set_ylim(0, 1.05)
b.set_xlabel("Spearman correlation of score with the order"); b.set_ylabel("P(median cutoff invalid)")
b.set_title("(b) all simulated problems", fontsize=10); b.legend(frameon=False, loc="upper right"); b.grid(ls=":", alpha=.5)
for ax in (a, b):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(ROOT / "figures" / "figure4_simulation.png", dpi=300)
