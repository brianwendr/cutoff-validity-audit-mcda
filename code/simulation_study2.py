#!/usr/bin/env python3
"""
Second Monte Carlo study (Section 9.6 of the manuscript): tolerance-level verdicts for a proposed cutoff.

Extends simulation_study.py with two new generator variables and the cutoff-level quantities of Section 5.
  smoothness  phi in {0, 0.8}: the noise eps_{qj} follows an AR(1) process along the order q with marginal N(0,1)
              (phi = 0 reproduces the independent noise of the first study);
  weights     equal, or w ~ Dirichlet(1,...,1) drawn per problem;
  plus the grid of Section 9.5: N+1 in {12,24}, m in {3,6}, delta in {0.1,0.25,0.5,1}, pi in {0,0.3}.
128 cells x 1000 problems = 128,000 problems.  Score: TOPSIS closeness with p = 2 (modal value); support
[a_q, b_q] = min/max over p in {1, 2, infinity}.  Symmetric tolerance tau = (b-a)/2; support-faithful asymmetric
tolerance (tm, tp) = (s-a, b-s).  Reported at tolerance level gamma = 1 and for the median cutoff.
Requires numpy.  Seed 20261021.  Writes outputs/simulation2_cells.csv, simulation2_summary.json,
logs/simulation2.log.
"""
from __future__ import annotations
import csv, json, itertools, time
from pathlib import Path
import numpy as np
from cutoff_margin import (certain_invalid_bands, possible_invalid_bands, in_bands, band_length, covered,
                           rho, phi)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"; LOGD = ROOT / "logs"
OUT.mkdir(exist_ok=True); LOGD.mkdir(exist_ok=True)

SEED = 20261021
REPS = 1000
GRID_N = [12, 24]
GRID_M = [3, 6]
GRID_DELTA = [0.1, 0.25, 0.5, 1.0]
GRID_PI = [0.0, 0.3]
GRID_AR = [0.0, 0.8]
GRID_W = ["equal", "dirichlet"]
OMEGAS = [0.05, 0.10]
EPS = 1e-12
INF = float("inf")


def topsis_all_norms(X, benefit, W):
    """X: (R,n,m) positive; W: (R,m) weights summing to 1; Euclidean column normalisation."""
    norm = np.sqrt((X ** 2).sum(axis=1, keepdims=True))
    V = W[:, None, :] * X / norm
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


def ar1_noise(rng, reps, n, m, phi_ar):
    eta = rng.standard_normal((reps, n, m))
    if phi_ar == 0.0:
        return eta
    eps = np.empty_like(eta); eps[:, 0] = eta[:, 0]
    sd = np.sqrt(1.0 - phi_ar ** 2)
    for q in range(1, n):
        eps[:, q] = phi_ar * eps[:, q - 1] + sd * eta[:, q]
    return eps


def radius_batch(S, tm, tp):
    """max_{i<j} (s_i - s_j)_+/(tm_i + tp_j) with x/0 = inf (x>0), 0/0 = 0."""
    n = S.shape[1]
    iu, ju = np.triu_indices(n, k=1)
    num = np.maximum(S[:, iu] - S[:, ju], 0.0)
    den = tm[:, iu] + tp[:, ju]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.where(num > 0, INF, 0.0))
    return ratio.max(axis=1)


def spearman_vs_order(S):
    R, n = S.shape
    ranks = S.argsort(axis=1).argsort(axis=1).astype(float)
    rq = np.arange(n, dtype=float) - (n - 1) / 2
    rc = ranks - ranks.mean(axis=1, keepdims=True)
    return (rc * rq).sum(axis=1) / np.sqrt((rc ** 2).sum(axis=1) * (rq ** 2).sum())


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rows = []
    chk = {"instances": 0, "rho_gt1_not_in_Z1": 0, "in_Z1_rho_lt1": 0, "phi_lt1_not_in_P1": 0, "in_P1_phi_gt1": 0,
           "valid_but_in_Z1": 0, "rho_cmed_gt_R": 0, "Z_not_subset_P_length": 0, "spectrum_vs_direct_mismatch": 0}
    for n, m, delta, pi, ar, wmode in itertools.product(GRID_N, GRID_M, GRID_DELTA, GRID_PI, GRID_AR, GRID_W):
        benefit = (np.arange(m) % 2 == 0)
        conflict = rng.random((REPS, m)) < pi
        direction = np.where(conflict, -1.0, 1.0)
        sign = np.where(benefit, 1.0, -1.0)[None, :]
        q = np.arange(n, dtype=float)[None, :, None]
        z = delta * q * (direction * sign)[:, None, :] + ar1_noise(rng, REPS, n, m, ar)
        X = np.exp(z)
        W = np.full((REPS, m), 1.0 / m) if wmode == "equal" else rng.dirichlet(np.ones(m), size=REPS)
        C = topsis_all_norms(X, benefit, W)
        S = C["p2"]
        a = np.minimum.reduce([C[k] for k in C]); b = np.maximum.reduce([C[k] for k in C])
        tau = np.maximum((b - a) / 2.0, EPS)
        tm_a, tp_a = S - a, b - S
        D = np.array([max(0.0, float((np.maximum.accumulate(s)[:-1] - s[1:]).max())) for s in S])
        valid = D <= EPS
        R_sym = radius_batch(S, tau, tau); R_asym = radius_batch(S, tm_a, tp_a)
        rho_sp = spearman_vs_order(S)
        acc = {k: [] for k in ("zshare", "pshare", "robust_share", "med_inv", "med_Z", "med_amb", "med_robust",
                                "rescue05", "rescue10", "local_ok_global_fail", "med_rho", "med_phi")}
        for r in range(REPS):
            s = S[r]; t = tau[r]
            rng_s = float(s.max() - s.min())
            Z = certain_invalid_bands(s, t, t, 1.0); P = possible_invalid_bands(s, t, t, 1.0)
            lz = band_length(Z, s.min(), s.max()); lp = band_length(P, s.min(), s.max())
            if lz > lp + 1e-12: chk["Z_not_subset_P_length"] += 1
            acc["zshare"].append(lz / rng_s); acc["pshare"].append((lp - lz) / rng_s); acc["robust_share"].append(1 - lp / rng_s)
            cm = float(np.median(s))
            inZ, inP = in_bands(cm, Z), in_bands(cm, P)
            rh, ph = rho(cm, s, t, t), phi(cm, s, t, t)
            direct_inv = bool(np.any((s >= cm)[:-1] & ~(s >= cm)[1:]))
            if in_bands(cm, certain_invalid_bands(s, t, t, 0.0)) != direct_inv: chk["spectrum_vs_direct_mismatch"] += 1
            chk["instances"] += 1
            if rh > 1 and not inZ: chk["rho_gt1_not_in_Z1"] += 1
            if inZ and rh < 1: chk["in_Z1_rho_lt1"] += 1
            if ph < 1 and not inP: chk["phi_lt1_not_in_P1"] += 1
            if inP and ph > 1: chk["in_P1_phi_gt1"] += 1
            if (not direct_inv) and inZ: chk["valid_but_in_Z1"] += 1
            if rh > R_sym[r] + 1e-9: chk["rho_cmed_gt_R"] += 1
            acc["med_inv"].append(direct_inv); acc["med_Z"].append(inZ)
            acc["med_amb"].append(inP and not inZ); acc["med_robust"].append(not inP)
            for om, key in zip(OMEGAS, ("rescue05", "rescue10")):
                lo, hi = cm - om * rng_s, cm + om * rng_s
                acc[key].append(not covered(lo, hi, P))
            acc["local_ok_global_fail"].append((rh <= 1) if R_sym[r] > 1 else np.nan)
            acc["med_rho"].append(rh); acc["med_phi"].append(ph)
        inv = ~valid
        glob_fail = R_sym > 1
        def mean(x):
            return float(np.nanmean(np.array(x, dtype=float)))
        rows.append({
            "n_alternatives": n, "m_criteria": m, "delta": delta, "pi_conflict": pi, "ar_phi": ar, "weights": wmode, "reps": REPS,
            "P_all_cutoffs_valid": round(float(valid.mean()), 4),
            "mean_spearman": round(float(rho_sp.mean()), 4),
            "P_R_sym_le_1_given_invalid": round(float((R_sym[inv] <= 1).mean()), 4) if inv.any() else "",
            "P_R_asym_le_1_given_invalid": round(float((R_asym[inv] <= 1).mean()), 4) if inv.any() else "",
            "P_R_asym_infinite_given_invalid": round(float(np.isinf(R_asym[inv]).mean()), 4) if inv.any() else "",
            "mean_Z1_share": round(mean(acc["zshare"]), 4), "mean_ambiguous_share": round(mean(acc["pshare"]), 4),
            "mean_robustly_valid_share": round(mean(acc["robust_share"]), 4),
            "P_median_invalid_nominal": round(mean(acc["med_inv"]), 4),
            "P_median_certainly_invalid": round(mean(acc["med_Z"]), 4),
            "P_median_ambiguous": round(mean(acc["med_amb"]), 4),
            "P_median_robustly_valid": round(mean(acc["med_robust"]), 4),
            "P_rescue_window_5pct": round(mean(acc["rescue05"]), 4), "P_rescue_window_10pct": round(mean(acc["rescue10"]), 4),
            "P_global_fail": round(float(glob_fail.mean()), 4),
            "P_median_rho_le_1_given_global_fail": round(mean(acc["local_ok_global_fail"]), 4) if glob_fail.any() else "",
            "mean_median_rho": round(mean(acc["med_rho"]), 4),
        })
        print(n, m, delta, pi, ar, wmode, rows[-1]["P_median_certainly_invalid"], rows[-1]["P_median_robustly_valid"], flush=True)
    with (OUT / "simulation2_cells.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = {"seed": SEED, "reps_per_cell": REPS, "cells": len(rows), "problems": len(rows) * REPS,
               "grid": {"n_alternatives": GRID_N, "m_criteria": GRID_M, "delta": GRID_DELTA, "pi_conflict": GRID_PI,
                        "ar_phi": GRID_AR, "weights": GRID_W, "rescue_windows": OMEGAS},
               "consistency_checks": chk, "seconds": round(time.time() - t0, 1)}
    (OUT / "simulation2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (LOGD / "simulation2.log").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
