#!/usr/bin/env python3
"""LaTeX table (reference design, by trend delta and smoothness) and pooled statistics from simulation2_cells.csv.
Requires pandas.  Writes tab_sim3.tex (copied next to the manuscript) and outputs/simulation2_pooled.json."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
d = pd.read_csv(ROOT / "outputs" / "simulation2_cells.csv")
ref = d[(d.n_alternatives == 12) & (d.m_criteria == 6) & (d.pi_conflict == 0.0) & (d.weights == "equal")].sort_values(["delta", "ar_phi"])
lines = [r"\begin{tabular}{@{}rrrrrrrrr@{}}", r"\toprule",
         r"$\delta$ & $\phi$ & Spearman & All valid & $R_\tau\le1\mid$inv. & Median in $Z_1$ & Median ambig. & Median in $\overline{P_1}$ & Rescue $\pm10\%$\\", r"\midrule"]
for _, r in ref.iterrows():
    lines.append(f"{r.delta:g} & {r.ar_phi:g} & {r.mean_spearman:.3f} & {r.P_all_cutoffs_valid:.3f} & {r.P_R_sym_le_1_given_invalid:.3f} & "
                 f"{r.P_median_certainly_invalid:.3f} & {r.P_median_ambiguous:.3f} & {r.P_median_robustly_valid:.3f} & {r.P_rescue_window_10pct:.3f}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(ROOT / "outputs" / "tab_sim3.tex").write_text("\n".join(lines) + "\n")
if len(sys.argv) > 1: Path(sys.argv[1]).write_text("\n".join(lines) + "\n")
pooled = {}
for key in ("ar_phi", "weights"):
    pooled[key] = d.groupby(key)[["P_all_cutoffs_valid", "P_R_sym_le_1_given_invalid", "P_median_invalid_nominal", "P_median_certainly_invalid",
              "P_median_ambiguous", "P_median_robustly_valid", "P_rescue_window_5pct", "P_rescue_window_10pct", "mean_Z1_share",
              "mean_ambiguous_share", "mean_robustly_valid_share"]].mean().round(4).to_dict("index")
pooled["overall"] = d[["P_all_cutoffs_valid", "P_R_sym_le_1_given_invalid", "P_R_asym_le_1_given_invalid", "P_median_invalid_nominal", "P_median_certainly_invalid",
                       "P_median_ambiguous", "P_median_robustly_valid", "P_rescue_window_5pct", "P_rescue_window_10pct", "P_global_fail",
                       "P_median_rho_le_1_given_global_fail"]].mean().round(4).to_dict()
# conditional probabilities must be pooled with the number of conditioning problems as weights, not averaged over cells
w_gf = d.P_global_fail * d.reps
pooled["overall"]["P_median_rho_le_1_given_global_fail"] = round(float((w_gf * d.P_median_rho_le_1_given_global_fail.fillna(0)).sum() / w_gf.sum()), 4)
w_inv = (1 - d.P_all_cutoffs_valid) * d.reps
for col in ("P_R_sym_le_1_given_invalid", "P_R_asym_le_1_given_invalid"):
    pooled["overall"][col] = round(float((w_inv * d[col].fillna(0)).sum() / w_inv.sum()), 4)
(ROOT / "outputs" / "simulation2_pooled.json").write_text(json.dumps(pooled, indent=2))
print(json.dumps(pooled["overall"], indent=1)); print("\n".join(lines))
