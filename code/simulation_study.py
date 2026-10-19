#!/usr/bin/env python3
"""
Monte Carlo study of cutoff validity for ordered TOPSIS decisions.

Question: in random ordered MCDA problems, how often do TOPSIS closeness scores fail to define
an upper-set (suffix) cutoff rule, how large are the invalid cutoff bands, how often are the
failures explainable by norm-sensitivity tolerances (R_tau <= 1), how often does the (sufficient)
matrix certificate hold, and how well do standard monotonicity diagnostics (Spearman correlation)
predict cutoff validity?

Generator (documented in the manuscript, Section 9.5):
  alternatives q = 0..N (operational order), m criteria, equal weights, half benefit / half cost
  (benefit/cost assigned by alternating index).
  log-score z_qj = delta * q * dir_j + eps_qj,  eps ~ N(0,1) iid, x_qj = exp(z_qj) > 0.
  dir_j = +1 (criterion improves along the order) except that each criterion independently
  "conflicts" with the order with probability pi (dir_j = -1).  Improving means increasing for
  benefit criteria and decreasing for cost criteria.
Scores: TOPSIS closeness C_p for p in {1, 2, infinity}; tolerance tau_q = (max_p C_p - min_p C_p)/2
(norm-sensitivity envelope, as in the UCI illustration).

Requires numpy.  Fixed seed.  Writes outputs/simulation_study_cells.csv,
outputs/simulation_study_summary.json, outputs/simulation_study_spearman_bins.csv and
logs/simulation_study.log.
"""
from __future__ import annotations
import csv, json, itertools, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs"
LOGD = ROOT / "logs"
OUT.mkdir(exist_ok=True); LOGD.mkdir(exist_ok=True)

SEED = 20261019
REPS = 2000
GRID_N = [6, 12, 24]            # number of alternatives (N+1)
GRID_M = [3, 6]                 # number of criteria
GRID_DELTA = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
GRID_PI = [0.0, 0.3]
NORMS = ["p1", "p2", "pinf"]
EPS = 1e-12

log_lines: list[str] = []
def log(msg: str) -> None:
    print(msg); log_lines.append(msg)


def topsis_all_norms(X: np.ndarray, benefit: np.ndarray) -> dict[str, np.ndarray]:
    """X: (R, n, m) positive; equal weights; Euclidean column normalisation; returns C for p=1,2,inf."""
    R, n, m = X.shape
    w = 1.0 / m
    norm = np.sqrt((X ** 2).sum(axis=1, keepdims=True))
    V = w * X / norm
    Aplus = np.where(benefit[None, None, :], V.max(axis=1, keepdims=True), V.min(axis=1, keepdims=True))
    Aminus = np.where(benefit[None, None, :], V.min(axis=1, keepdims=True), V.max(axis=1, keepdims=True))
    dP = np.abs(V - Aplus); dM = np.abs(V - Aminus)
    out = {}
    for name, p in (("p1", 1.0), ("p2", 2.0), ("pinf", np.inf)):
        if np.isinf(p):
            dp, dm = dP.max(axis=2), dM.max(axis=2)
        else:
            dp, dm = (dP ** p).sum(axis=2) ** (1 / p), (dM ** p).sum(axis=2) ** (1 / p)
        out[name] = dm / (dp + dm)
    return out


def spectrum_union_length(s: np.ndarray) -> tuple[float, list[tuple[float, float]]]:
    """Compressed spectrum (Theorem 1) merged into disjoint intervals; returns total length."""
    iv = []
    M = s[0]
    for j in range(1, len(s)):
        if M > s[j]:
            iv.append((s[j], M))
        M = max(M, s[j])
    if not iv:
        return 0.0, []
    iv.sort()
    merged = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return float(sum(b - a for a, b in merged)), [(a, b) for a, b in merged]


def in_union(c: float, merged: list[tuple[float, float]]) -> bool:
    return any(a < c <= b for a, b in merged)


def direct_invalid(s: np.ndarray, c: float) -> bool:
    acc = s >= c
    return bool(np.any(acc[:-1] & ~acc[1:]))          # accepted earlier, rejected later


def weighted_radius(S: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """S, tau: (R, n).  max_{i<j} max(s_i - s_j, 0)/(tau_i + tau_j)  (Proposition 3)."""
    n = S.shape[1]
    iu, ju = np.triu_indices(n, k=1)
    diff = np.maximum(S[:, iu] - S[:, ju], 0.0)
    return (diff / (tau[:, iu] + tau[:, ju])).max(axis=1)


def spearman_vs_order(S: np.ndarray) -> np.ndarray:
    R, n = S.shape
    ranks = S.argsort(axis=1).argsort(axis=1).astype(float)
    q = np.arange(n, dtype=float)
    rq = q - q.mean()
    rc = ranks - ranks.mean(axis=1, keepdims=True)
    return (rc * rq).sum(axis=1) / np.sqrt((rc ** 2).sum(axis=1) * (rq ** 2).sum())


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    cell_rows = []
    pooled = {k: {"rho": [], "med_inv": [], "inv_share": [], "valid": []} for k in NORMS}
    totals = {"matrix_cert_violations": 0, "spectrum_vs_direct_mismatches": 0, "instances": 0,
              "matrix_cert_true": 0, "matrix_cert_true_and_p2_valid": 0, "p2_valid": 0}
    for n, m, delta, pi in itertools.product(GRID_N, GRID_M, GRID_DELTA, GRID_PI):
        benefit = (np.arange(m) % 2 == 0)
        conflict = rng.random((REPS, m)) < pi
        direction = np.where(conflict, -1.0, 1.0)                      # +1: improves along the order
        sign = np.where(benefit, 1.0, -1.0)[None, :]                    # +1 benefit: improving = increasing
        q = np.arange(n, dtype=float)[None, :, None]
        z = delta * q * (direction * sign)[:, None, :] + rng.standard_normal((REPS, n, m))
        X = np.exp(z)
        # matrix certificate (Theorem 3): benefit nondecreasing, cost nonincreasing along q
        dX = np.diff(X, axis=1)
        ok = np.where(benefit[None, None, :], dX >= 0, dX <= 0).all(axis=(1, 2))
        C = topsis_all_norms(X, benefit)
        tau = (np.maximum.reduce([C[k] for k in NORMS]) - np.minimum.reduce([C[k] for k in NORMS])) / 2.0
        tau = np.maximum(tau, EPS)
        S2 = C["p2"]
        D2 = np.array([max(0.0, float((np.maximum.accumulate(s)[:-1] - s[1:]).max())) for s in S2])
        valid2 = D2 <= EPS
        totals["instances"] += REPS
        totals["matrix_cert_true"] += int(ok.sum())
        totals["matrix_cert_true_and_p2_valid"] += int((ok & valid2).sum())
        totals["p2_valid"] += int(valid2.sum())
        for k in NORMS:
            S = C[k]
            D = np.array([max(0.0, float((np.maximum.accumulate(s)[:-1] - s[1:]).max())) for s in S])
            valid = D <= EPS
            if np.any(ok & ~valid):
                totals["matrix_cert_violations"] += int((ok & ~valid).sum())
            Rt = weighted_radius(S, tau)
            rho = spearman_vs_order(S)
            shares, med_inv, tier_inv = [], [], []
            for s in S:
                rng_s = float(s.max() - s.min())
                L, merged = spectrum_union_length(s)
                shares.append(L / rng_s if rng_s > 0 else 0.0)
                c_med = float(np.median(s))
                inv_spec = in_union(c_med, merged)
                if inv_spec != direct_invalid(s, c_med):
                    totals["spectrum_vs_direct_mismatches"] += 1
                med_inv.append(inv_spec)
                c1, c2 = np.quantile(s, [1 / 3, 2 / 3])
                tier_inv.append(in_union(float(c1), merged) or in_union(float(c2), merged))
            shares = np.array(shares); med_inv = np.array(med_inv); tier_inv = np.array(tier_inv)
            inv_seq = ~valid
            cell_rows.append({
                "n_alternatives": n, "m_criteria": m, "delta": delta, "pi_conflict": pi, "norm": k,
                "reps": REPS,
                "P_all_cutoffs_valid": round(float(valid.mean()), 4),
                "P_matrix_certificate": round(float(ok.mean()), 4),
                "mean_invalid_band_share": round(float(shares.mean()), 4),
                "P_median_cutoff_invalid": round(float(med_inv.mean()), 4),
                "P_tercile_tiers_invalid": round(float(tier_inv.mean()), 4),
                "median_R_tau_given_invalid": round(float(np.median(Rt[inv_seq])), 4) if inv_seq.any() else "",
                "P_R_tau_le_1_given_invalid": round(float((Rt[inv_seq] <= 1).mean()), 4) if inv_seq.any() else "",
                "mean_spearman": round(float(rho.mean()), 4),
            })
            pooled[k]["rho"].append(rho); pooled[k]["med_inv"].append(med_inv)
            pooled[k]["inv_share"].append(shares); pooled[k]["valid"].append(valid)
    # Spearman bins (pooled over the whole grid, per norm)
    bins = [(-1.01, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]
    bin_rows = []
    for k in NORMS:
        rho = np.concatenate(pooled[k]["rho"]); mi = np.concatenate(pooled[k]["med_inv"])
        sh = np.concatenate(pooled[k]["inv_share"]); va = np.concatenate(pooled[k]["valid"])
        for lo, hi in bins:
            sel = (rho >= lo) & (rho < hi)
            if sel.sum() == 0:
                continue
            bin_rows.append({"norm": k, "rho_lo": max(lo, -1.0), "rho_hi": min(hi, 1.0), "instances": int(sel.sum()),
                             "P_all_cutoffs_valid": round(float(va[sel].mean()), 4),
                             "P_median_cutoff_invalid": round(float(mi[sel].mean()), 4),
                             "mean_invalid_band_share": round(float(sh[sel].mean()), 4)})
    with (OUT / "simulation_study_cells.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys())); w.writeheader(); w.writerows(cell_rows)
    with (OUT / "simulation_study_spearman_bins.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys())); w.writeheader(); w.writerows(bin_rows)
    summary = {"seed": SEED, "reps_per_cell": REPS, "cells": len(GRID_N) * len(GRID_M) * len(GRID_DELTA) * len(GRID_PI),
               "grid": {"n_alternatives": GRID_N, "m_criteria": GRID_M, "delta": GRID_DELTA, "pi_conflict": GRID_PI},
               **totals, "seconds": round(time.time() - t0, 1)}
    (OUT / "simulation_study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary))
    (LOGD / "simulation_study.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
