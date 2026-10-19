#!/usr/bin/env python3
"""Figure 5: tolerance-level map (c, gamma) and signed margin profile for the UCI geometry profiles.
Requires matplotlib and numpy; run after uci_margin.py.  Reads outputs/norm_specific_open_data_scores_wide.csv."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from cutoff_margin import certain_invalid_bands, possible_invalid_bands, in_bands, margin

ROOT = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader((ROOT / "outputs" / "norm_specific_open_data_scores_wide.csv").open()))
S = {k: np.array([float(r[k]) for r in rows]) for k in ("p1", "p2", "pinf")}
a = np.minimum.reduce(list(S.values())); b = np.maximum.reduce(list(S.values())); tau = (b - a) / 2
C0 = 0.55

s = S["p2"]
cs = np.linspace(0.28, 0.72, 441); gs = np.linspace(0.0, 6.0, 241)
M = np.zeros((len(gs), len(cs)))
for i, g in enumerate(gs):
    Z = certain_invalid_bands(s, tau, tau, float(g)); P = possible_invalid_bands(s, tau, tau, float(g))
    for j, c in enumerate(cs):
        M[i, j] = 2 if in_bands(c, Z) else (1 if in_bands(c, P) else 0)

fig, (A, B) = plt.subplots(1, 2, figsize=(10.0, 3.9), gridspec_kw={"width_ratios": [1.05, 1.0]})
cmap = ListedColormap(["#cfe8cf", "#f6e3a1", "#e9a29a"])
A.pcolormesh(cs, gs, M, cmap=cmap, shading="auto", rasterized=True)
A.axvline(C0, color="k", lw=1.0, ls="--"); A.axhline(1.0, color="k", lw=1.0, ls=":")
A.text(C0 + 0.004, 5.55, r"$c_0$", fontsize=9); A.text(0.285, 1.08, r"$\gamma=1$", fontsize=9)
A.text(0.44, 0.35, "certainly\ninvalid", fontsize=8, ha="center", color="#7a1f16")
A.text(0.31, 0.3, "robustly\nvalid", fontsize=7.5, ha="center", color="#235023")
A.text(0.67, 0.3, "robustly valid", fontsize=7.5, ha="center", color="#235023")
A.text(0.60, 3.6, "ambiguous", fontsize=9, ha="center", color="#6b5a00")
A.set_xlabel("Cutoff $c$"); A.set_ylabel(r"Tolerance level $\gamma$")
A.set_title(r"(a) $p=2$: $Z_\gamma$ (red), $P_\gamma\setminus Z_\gamma$ (amber)", fontsize=10)
A.set_xlim(cs[0], cs[-1]); A.set_ylim(0, 6)

grid = np.linspace(0.28, 0.72, 881)
for k, lab, ls in (("p1", r"$p=1$", "-"), ("p2", r"$p=2$", "--"), ("pinf", r"$p=\infty$", ":")):
    B.plot(grid, [max(-5.0, min(3.0, margin(c, S[k], tau, tau))) for c in grid], ls=ls, lw=1.8, label=lab)
B.axhspan(-1, 1, color="0.9", zorder=0); B.axhline(0, color="k", lw=0.8); B.axvline(C0, color="k", lw=1.0, ls="--")
B.set_ylim(-5.2, 3.2); B.set_xlim(0.28, 0.72)
B.set_xlabel("Cutoff $c$"); B.set_ylabel(r"Signed margin $m(c)=\varphi(c)-\rho(c)$")
B.set_title(r"(b) margin profile (clipped to $[-5,3]$; shaded: $|m|\leq 1$)", fontsize=10)
B.legend(frameon=False, loc="lower right"); B.grid(ls=":", alpha=.5)
for ax in (A, B):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(ROOT / "figures" / "figure5_cutoff_margin.png", dpi=300)
print("ok")
