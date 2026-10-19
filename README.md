# Supplementary material: replication package

Manuscript: *A tolerance-weighted cutoff-validity audit for ordered TOPSIS and VIKOR decisions*.

## Contents

- `code/run_replication.py`: main script (Python standard library only; matplotlib optional for Figure 3). Regenerates all synthetic and open-data outputs.
- `code/extra_checks.py`: independent verification and baseline comparison (requires `numpy` and `scipy`; fixed seed 20261016). Does not alter the outputs of `run_replication.py`.
- `code/make_figure2.py`: regenerates Figure 2 from `outputs/norm_specific_open_data_scores_wide.csv` (requires `matplotlib`).
- `code/simulation_study.py`: Monte Carlo study of cutoff validity in random ordered TOPSIS problems (Section 9.5) (requires `numpy`; seed 20261019; about 70 s).
- `code/make_figure4.py`: regenerates the simulation figure (Figure 5) from the simulation outputs (requires `matplotlib`).
- `code/cutoff_margin.py`: core library for Section 5 of the manuscript (certainly/possibly invalid cutoff sets `Z_gamma`, `P_gamma`, cutoff-level radii `rho(c)`, `phi(c)`, signed margin `m(c)`, global radius, critical cutoff, exact margin-optimal cutoff in a window; numpy only).
- `code/verify_margin.py`: independent verification of Section 5 by corner enumeration, bisection, linear programming and dense grids (numpy, scipy; seed 20261020; about 2.5 min).
- `code/uci_margin.py`: applies Section 5 to the UCI geometry profiles (Tables 9-10, Figure 4 data).
- `code/make_figure5.py`: regenerates the tolerance-level map and margin profile (Figure 4; file `figures/figure5_cutoff_margin.png`).
- `code/simulation_study2.py`: second Monte Carlo study (noise smoothness, random weights, tolerance-level verdicts; numpy; seed 20261021; about 25 s).
- `code/make_tables2.py`: LaTeX table and pooled statistics of the second study (pandas).
- `data/uci_geometry_profiles.csv`: ordered geometry-profile abstraction derived from the public UCI Energy Efficiency design grid.
- `outputs/`: generated CSV and JSON files (tables, cutoff-validity spectra, repair radii, tolerance inputs, sensitivity, verification summary, baseline comparison).
- `figures/`: Figure 1 (audit workflow, TikZ source `figure1_audit_workflow.tex` plus compiled PDF/PNG), Figure 2 (open-data TOPSIS scores), Figure 3 (weighted repair sensitivity).
- `logs/trace.log`, `logs/extra_checks.log`: run logs.

## How to run

```bash
python code/run_replication.py     # main outputs (standard library only)
python code/extra_checks.py        # LP / exhaustive / randomised checks and baseline table (numpy, scipy)
python code/make_figure2.py        # Figure 2 (matplotlib)
python code/simulation_study.py    # simulation study, Section 9.5 (numpy)
python code/make_figure4.py        # simulation figure (matplotlib)
cd code                            # the Section 5 scripts import cutoff_margin.py from their own directory
python verify_margin.py            # independent verification of Section 5 (numpy, scipy)
python uci_margin.py               # cutoff-level audit of the UCI profiles
python make_figure5.py             # tolerance-level map (matplotlib)
python simulation_study2.py        # second simulation study
python make_tables2.py             # LaTeX table and pooled statistics (pandas)
```

## What each output supports in the manuscript

| Manuscript item | File |
|---|---|
| Section 9.1 synthetic examples, Table 3 | `table1_base_topsis.csv`, `table2_modified_topsis.csv`, `table3_LHL_obstruction.csv`, `cutoff_validity_spectrum_LHL.csv`, `repair_radius_synthetic.csv`, `weighted_repair_radius_synthetic.csv`, `triangular_fuzzy_lhl_tolerances.csv` |
| Section 9.2, Tables 4-5, Figure 2 | `triangular_fuzzy_open_data_scores.csv`, `norm_specific_open_data_scores_wide.csv`, `open_data_geometry_scores.csv` |
| Table 6 | `open_data_cutoff_validity_spectrum.csv`, `repair_radius_open_data.csv`, `weighted_repair_radius_open_data.csv` |
| Table 7, Figure 3, Table 8 | `tolerance_sensitivity.csv`, `tolerance_scenario_summary.csv` |
| Section 9.4, Table 11 (comparison with standard diagnostics) | `baseline_comparison.csv` |
| Section 9.5 (first simulation study), Tables 12-13, Figure 5 | `simulation_study_cells.csv` (all 72 cells x 3 norms), `simulation_study_spearman_bins.csv`, `simulation_study_summary.json`, `logs/simulation_study.log` |
| Section 9.3 (cutoff-level audit of the UCI profiles), Tables 9-10, Figure 4 | `uci_margin_summary.json` (all quantities, symmetric and asymmetric tolerances), `uci_margin_quota.csv`, `uci_margin_profile.csv`, `uci_margin_phase.csv` |
| Section 9.6 (second simulation study), Table 14 | `simulation2_cells.csv` (128 cells), `simulation2_summary.json`, `simulation2_pooled.json`, `tab_sim3.tex`, `logs/simulation2.log` |
| Section 9.7 (independent verification) | `extra_checks_summary.json`, `logs/extra_checks.log`, `margin_verification_summary.json`, `logs/margin_verification.log` |

## Verification summary (`extra_checks.py`)

- Propositions 2-3: 3000 random instances (2594 with an inversion); maximum absolute difference between the closed forms and a linear-programming solution is 2.1e-14 (weighted) and 1.1e-16 (uniform).
- Theorem 1: 4000 integer-valued sequences, 84,000 cutoff cells, 0 mismatches.
- Theorems 2-3 and the stability radius: 1500 random matrices with p in {1, 2, infinity}, 0 violations.

## Verification summary for Section 5 (`verify_margin.py`)

- Oracle independent of the closed forms: all 2^(N+1) corner score vectors of the admissible box, direct suffix test. Membership in `Z_gamma` and `P_gamma`: 36,000 random (score, tolerance, level, cutoff) combinations, 0 mismatches (also 36,000 with one-sided zero tolerances, 0 mismatches). `rho(c)` and `phi(c)` against bisection on the oracle: relative difference 5.1e-15 and 4.4e-15 (1500 instances; 5.8e-15 with zero-sided tolerances).
- Global radius against a linear-programming solution of asymmetric L_inf isotonic repair: 3000 instances, maximum difference 1.7e-14; `rho(c*) = R` at the critical cutoff and `rho <= R` everywhere, 0 violations.
- `Z_gamma` subset of `P_gamma`, monotonicity in gamma, `Z_0 = P_0` = spectrum of Theorem 1, `Z_gamma` empty iff gamma >= R: 0 violations in 36,000 cutoff checks.
- Exact margin-optimal cutoff never below a 4001-point grid search (400 windows).

## Simulation design (`simulation_study.py`)

N+1 in {6,12,24} alternatives, m in {3,6} criteria with equal weights (benefit and cost alternating), log-entries z = delta*q*d_j*sigma_j + eps with eps ~ N(0,1), x = exp(z); each criterion conflicts with the order (d_j = -1) with probability pi in {0, 0.3}; delta in {0, 0.1, 0.25, 0.5, 1, 2}. 72 cells x 2000 problems = 144,000 problems; scores are TOPSIS closeness for p = 1, 2, infinity; tolerances are the norm-envelope half-widths. Recorded: all-cutoff validity, invalid share of the score range, median-cutoff and tercile-tier validity, matrix certificate, Spearman correlation, and whether R_tau <= 1 for invalid sequences. Consistency checks: spectrum test vs direct suffix test (0 mismatches in 432,000 sequences) and Theorem 3 (0 violations).

## Second simulation design (`simulation_study2.py`)

N+1 in {12,24}, m in {3,6}, delta in {0.1,0.25,0.5,1}, pi in {0,0.3}, noise smoothness phi in {0,0.8} (AR(1) along the order, marginal N(0,1)), weights equal or Dirichlet(1,...,1) per problem: 128 cells x 1000 = 128,000 problems. Audited score: TOPSIS p=2 (modal value); support = min/max over p in {1,2,infinity}; symmetric tolerance (b-a)/2 and support-faithful asymmetric tolerance (s-a, b-s). Evaluated at gamma = 1 for the median cutoff: nominal validity, certainly invalid (`Z_1`), ambiguous, robustly valid (outside `P_1`), shares of the score range, `R_tau <= 1` given invalid, whether `rho(c_med) <= 1` when `R_tau > 1`, and whether a robustly valid cutoff exists within +-5% / +-10% of the score range. Six internal consistency relations are checked in every problem (0 violations in 128,000). Pooled conditional probabilities are weighted by the number of conditioning problems (`make_tables2.py`), not averaged over cells.

## Conventions

- Tolerances are instantiated from triangular fuzzy scores `(a_q, s_q, b_q)` by `tau_q = (b_q - a_q)/2`. In the UCI illustration the p=2 TOPSIS score is the modal value and the support is the min/max over p=1, p=2, p=infinity (norm-sensitivity envelope: spread across plausible norms as a proxy for representational uncertainty).
- Sentinel: for `Q = {0,...,N}`, `T = N+1` denotes the empty acceptance set. For p=infinity at `c0 = 0.55`, `T = 12`: no alternative reaches `c0`.
- The worst tolerance-weighted pair is `(4,7)` for p=1, p=2 and p=infinity (`weighted_repair_radius_open_data.csv`); it can be verified directly from `norm_specific_open_data_scores_wide.csv` and the tolerances.
- No proprietary data are used.

## Notes

- The numerical outputs of `run_replication.py` (Sections 9.1-9.2 and the theorem checks) do not depend on the later scripts; the Section 5 scripts (`cutoff_margin.py`, `verify_margin.py`, `uci_margin.py`, `make_figure5.py`) and the second simulation (`simulation_study2.py`, `make_tables2.py`) add the cutoff-level results of Sections 9.3 and 9.6.
- All random number generators use fixed seeds (listed above), so the outputs can be regenerated exactly.
