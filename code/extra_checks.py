#!/usr/bin/env python3
"""
Supplementary independent checks for:
A tolerance-weighted cutoff-validity audit for ordered TOPSIS and VIKOR decisions

This script is separate from run_replication.py and does not alter its outputs.
It requires numpy and scipy (run_replication.py itself is standard-library only).

It performs four things:
  1. LP verification of Propositions 2 and 3 (closed-form repair radii) on random instances.
  2. Exhaustive-grid verification of Theorem 1 (cutoff-validity spectrum) on integer-valued sequences.
  3. Randomised verification of Theorems 2 and 3 (TOPSIS ratio equivalence and matrix certificate).
  4. A baseline comparison on the UCI geometry-profile and low-high-low examples:
     rank correlation, count of inverted pairs, L2 isotonic (PAVA) fit, versus the audit quantities.

Outputs are written to outputs/ and logs/extra_checks.log. A fixed seed is used.
"""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.stats import kendalltau, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs"
LOG = ROOT / "logs" / "extra_checks.log"
OUT.mkdir(exist_ok=True)
(ROOT / "logs").mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import run_replication as rr  # noqa: E402  (reuses topsis, spectrum, repair functions)

RNG = np.random.default_rng(20261016)
log_lines: list[str] = []


def log(msg: str) -> None:
    print(msg)
    log_lines.append(msg)


# ---------------------------------------------------------------------------
# 1. LP verification of Propositions 2 and 3
# ---------------------------------------------------------------------------
def lp_weighted_radius(s: np.ndarray, tau: np.ndarray) -> float:
    """min lambda s.t. |s_q - y_q| <= lambda*tau_q and y nondecreasing (LP in (y, lambda))."""
    n = len(s)
    c = np.zeros(n + 1)
    c[-1] = 1.0
    A, b = [], []
    for q in range(n):
        row = np.zeros(n + 1); row[q] = 1.0; row[-1] = -tau[q]; A.append(row); b.append(s[q])      # y_q - s_q <= lam*tau_q
        row = np.zeros(n + 1); row[q] = -1.0; row[-1] = -tau[q]; A.append(row); b.append(-s[q])    # s_q - y_q <= lam*tau_q
    for q in range(n - 1):
        row = np.zeros(n + 1); row[q] = 1.0; row[q + 1] = -1.0; A.append(row); b.append(0.0)       # y_q <= y_{q+1}
    bounds = [(None, None)] * n + [(0, None)]
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")
    assert res.status == 0, res.message
    return float(res.fun)


def check_repair_radii(n_inst: int = 3000) -> dict:
    max_err_w = 0.0
    max_err_u = 0.0
    n_invalid = 0
    for _ in range(n_inst):
        n = int(RNG.integers(2, 16))
        kind = RNG.integers(0, 3)
        if kind == 0:
            s = RNG.random(n)
        elif kind == 1:
            s = RNG.integers(0, 6, size=n).astype(float) / 5.0       # many ties
        else:
            s = np.sort(RNG.random(n)) + 0.3 * RNG.standard_normal(n) * (RNG.random(n) < 0.3)
        tau = RNG.uniform(0.01, 1.0, size=n)
        closed_w = rr.weighted_monotone_repair_radius(list(s), list(tau))
        closed_u = rr.monotone_repair_radius(list(s))
        lp_w = lp_weighted_radius(s, tau)
        lp_u = lp_weighted_radius(s, np.ones(n))
        max_err_w = max(max_err_w, abs(closed_w - lp_w))
        max_err_u = max(max_err_u, abs(closed_u - lp_u))
        if closed_u > 1e-12:
            n_invalid += 1
    return {"instances": n_inst, "instances_with_inversion": n_invalid,
            "max_abs_error_weighted_closed_form_vs_LP": max_err_w,
            "max_abs_error_uniform_closed_form_vs_LP": max_err_u}


# ---------------------------------------------------------------------------
# 2. Exhaustive-grid verification of Theorem 1
# ---------------------------------------------------------------------------
def is_suffix_at(s: list[int], c2: int) -> bool:
    """cutoff c = c2/2 on integer scores; accepted set {q : 2*s_q >= c2} must be a suffix."""
    acc = [2 * v >= c2 for v in s]
    return all(acc[i] <= acc[i + 1] for i in range(len(acc) - 1))


def in_spectrum(s: list[int], c2: int) -> bool:
    M = s[0]
    for j in range(1, len(s)):
        if M > s[j] and 2 * s[j] < c2 <= 2 * M:
            return True
        M = max(M, s[j])
    return False


def check_theorem1(n_inst: int = 4000) -> dict:
    mismatches = 0
    cells = 0
    for _ in range(n_inst):
        n = int(RNG.integers(2, 12))
        s = [int(v) for v in RNG.integers(0, 8, size=n)]
        for c2 in range(-2, 2 * 8 + 3):       # all half-integer cutoffs, including outside range
            cells += 1
            if (not is_suffix_at(s, c2)) != in_spectrum(s, c2):
                mismatches += 1
    return {"sequences": n_inst, "cutoff_cells_checked": cells, "mismatches": mismatches}


# ---------------------------------------------------------------------------
# 3. Theorems 2 and 3
# ---------------------------------------------------------------------------
def random_matrix(N: int, m: int, monotone_direction: bool):
    benefit = [bool(b) for b in RNG.integers(0, 2, size=m)]
    X = np.zeros((N + 1, m))
    for j in range(m):
        base = RNG.uniform(1.0, 5.0)
        if monotone_direction:
            inc = RNG.uniform(0.0, 1.0, size=N)
            col = base + np.concatenate([[0.0], np.cumsum(inc)])
            X[:, j] = col if benefit[j] else col[::-1]          # benefit ascending, cost descending along q
        else:
            X[:, j] = RNG.uniform(1.0, 9.0, size=N + 1)
    return X, benefit


def check_topsis_theorems(n_inst: int = 1500) -> dict:
    res = {"ratio_equiv_mismatch": 0, "matrix_cert_violation": 0, "stability_radius_violation": 0,
           "random_instances": n_inst}
    for _ in range(n_inst):
        N = int(RNG.integers(2, 10)); m = int(RNG.integers(2, 6))
        w = RNG.dirichlet(np.ones(m))
        for p in (1.0, 2.0, math.inf):
            # Theorem 2: C nondecreasing  <=>  ratio nondecreasing (arbitrary matrices)
            X, ben = random_matrix(N, m, monotone_direction=False)
            rows = rr.topsis(X.tolist(), list(w), ben, p)
            C = [r["C"] for r in rows]; R = [r["ratio"] for r in rows]
            if rr.is_nondecreasing(C, 1e-12) != rr.is_nondecreasing(R, 1e-9):
                # allow borderline ties only
                if min(abs(C[i + 1] - C[i]) for i in range(N)) > 1e-9:
                    res["ratio_equiv_mismatch"] += 1
            # Theorem 3: monotone-direction matrix => C nondecreasing
            Xm, benm = random_matrix(N, m, monotone_direction=True)
            Cm = [r["C"] for r in rr.topsis(Xm.tolist(), list(w), benm, p)]
            if not rr.is_nondecreasing(Cm, 1e-12):
                res["matrix_cert_violation"] += 1
            # Stability radius: perturb every entry by at most rho = mu/2 (strictly inside) and re-check
            gaps = []
            for j in range(m):
                col = Xm[:, j]
                gaps += list((col[1:] - col[:-1]) if benm[j] else (col[:-1] - col[1:]))
            mu = min(gaps)
            if mu > 1e-6:
                eps = 0.999 * mu / 2.0
                Xp = Xm + RNG.uniform(-eps, eps, size=Xm.shape)
                Cp = [r["C"] for r in rr.topsis(Xp.tolist(), list(w), benm, p)]
                if not rr.is_nondecreasing(Cp, 1e-12):
                    res["stability_radius_violation"] += 1
    return res


# ---------------------------------------------------------------------------
# 4. Baseline comparison
# ---------------------------------------------------------------------------
def pava_l2(y: list[float]) -> list[float]:
    """Pool-adjacent-violators: L2 isotonic regression (unit weights)."""
    blocks = [[v, 1] for v in y]            # [mean, size]
    out: list[list[float]] = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0] + 1e-15:
            m2, n2 = out.pop(); m1, n1 = out.pop()
            out.append([(m1 * n1 + m2 * n2) / (n1 + n2), n1 + n2])
    fit = []
    for mean, size in out:
        fit += [mean] * int(size)
    return fit


def baseline_row(label: str, s: list[float], tau: list[float] | None, c0: float) -> dict:
    n = len(s)
    q = list(range(n))
    rho = spearmanr(q, s).statistic
    ktau = kendalltau(q, s).statistic
    inv = sum(1 for i in range(n) for j in range(i + 1, n) if s[i] > s[j])
    pairs = n * (n - 1) // 2
    fit = pava_l2(s)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(s, fit)) / n)
    linf_fit = max(abs(a - b) for a, b in zip(s, fit))
    spec = rr.cutoff_validity_spectrum(s)
    # length of the union of invalid intervals (merge)
    ivs = sorted((r["invalid_cutoff_open_lower"], r["invalid_cutoff_closed_upper"]) for r in spec)
    total = 0.0; cur = None
    for a, b in ivs:
        if cur is None: cur = [a, b]
        elif a <= cur[1]: cur[1] = max(cur[1], b)
        else: total += cur[1] - cur[0]; cur = [a, b]
    if cur: total += cur[1] - cur[0]
    inside = any(a < c0 <= b for a, b in ivs)
    D = rr.max_inversion_depth(s)
    return {
        "case": label,
        "spearman_rho_vs_order": round(rho, 4),
        "kendall_tau_vs_order": round(ktau, 4),
        "inverted_pairs": f"{inv}/{pairs}",
        "L2_isotonic_RMSE": round(rmse, 4),
        "L2_isotonic_max_abs_dev": round(linf_fit, 4),
        "max_inversion_depth_D": round(D, 4),
        "R_inf_uniform": round(D / 2, 4),
        "R_tau_weighted": "" if tau is None else round(rr.weighted_monotone_repair_radius(s, tau), 4),
        "invalid_cutoff_band_length": round(total, 4),
        "score_range": round(max(s) - min(s), 4),
        "c0": c0,
        "c0_in_invalid_spectrum": inside,
    }


def baseline_comparison() -> list[dict]:
    wide = list(csv.DictReader((OUT / "norm_specific_open_data_scores_wide.csv").open()))
    fuzzy = list(csv.DictReader((OUT / "triangular_fuzzy_open_data_scores.csv").open()))
    tau = [float(r["tau_q"]) for r in fuzzy]
    rows = []
    for key, name in (("p1", "UCI p=1"), ("p2", "UCI p=2"), ("pinf", "UCI p=infinity")):
        s = [float(r[key]) for r in wide]
        rows.append(baseline_row(name, s, tau, 0.55))
    lhl_tau = [0.05, 0.06, 0.07]
    rows.append(baseline_row("Low-high-low (0.30,0.60,0.40)", [0.30, 0.60, 0.40], lhl_tau, 0.55))
    return rows


def main() -> None:
    results = {}
    log("1. LP check of Propositions 2-3 ...")
    results["propositions_2_3_vs_LP"] = check_repair_radii()
    log(json.dumps(results["propositions_2_3_vs_LP"]))
    log("2. Exhaustive-grid check of Theorem 1 ...")
    results["theorem_1_grid"] = check_theorem1()
    log(json.dumps(results["theorem_1_grid"]))
    log("3. Randomised check of Theorems 2-3 ...")
    results["theorems_2_3_random"] = check_topsis_theorems()
    log(json.dumps(results["theorems_2_3_random"]))
    log("4. Baseline comparison ...")
    rows = baseline_comparison()
    with (OUT / "baseline_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        log(json.dumps(r))
    results["baseline_comparison_rows"] = len(rows)
    with (OUT / "extra_checks_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with LOG.open("w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
