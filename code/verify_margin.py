#!/usr/bin/env python3
"""
Independent verification of the cutoff-level results of Section 5 (numpy + scipy; fixed seed 20261020).

Oracle (shares no formula with cutoff_margin.py): for alternative q the score may be moved anywhere in the
closed band [l_q, u_q]; whether cutoff c accepts q depends only on the side of c, so every acceptance pattern
reachable inside the box is reached by some corner y in {l_q, u_q}^{N+1}.  We enumerate all corners, test
"accepted set is a suffix" directly, and compare:
  (V1) c in Z_gamma  <=>  no corner makes c valid;  c in P_gamma  <=>  some corner makes c invalid
  (V2) rho(c) = inf{gamma : some corner makes c valid},  phi(c) = inf{gamma : some corner makes c invalid}  (bisection)
  (V3) R = max_{i<j}(s_i-s_j)_+/(tm_i+tp_j) equals the LP optimum of asymmetric L_inf isotonic repair,
       rho(c*) = R at the critical cutoff, rho(c) <= R everywhere, and max_c rho(c) = R on a dense grid
  (V4) Z_gamma subset P_gamma, nesting in gamma, Z_0 = P_0 = spectrum of Theorem 1 (direct suffix test),
       Z_gamma empty iff gamma >= R
  (V5) the exact maximum-margin cutoff in a window equals (or beats) a dense-grid search
Writes outputs/margin_verification_summary.json and logs/margin_verification.log.
"""
from __future__ import annotations
import json, itertools, time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

from cutoff_margin import (certain_invalid_bands, possible_invalid_bands, in_bands, rho, phi, margin,
                           global_radius, critical_cutoff, max_margin_cutoff, band_ends)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"; LOGD = ROOT / "logs"
OUT.mkdir(exist_ok=True); LOGD.mkdir(exist_ok=True)
rng = np.random.default_rng(20261020)
ZERO_SIDES = False
lines: list[str] = []
def log(m):
    print(m); lines.append(m)


def is_suffix(acc):
    return not np.any(acc[:-1] & ~acc[1:])


_patterns = {}
def patterns(n):
    if n not in _patterns:
        _patterns[n] = np.array(list(itertools.product([False, True], repeat=n)))
    return _patterns[n]


def corner_oracle(c, s, tm, tp, gamma):
    """returns (exists corner valid, exists corner invalid)"""
    l, u = band_ends(s, tm, tp, gamma)
    pat = patterns(len(s))
    Y = np.where(pat, u[None, :], l[None, :])
    acc = Y >= c
    bad = (acc[:, :-1] & ~acc[:, 1:]).any(axis=1)          # invalid corners
    return bool((~bad).any()), bool(bad.any())


def random_instance(n):
    kind = rng.integers(0, 3)
    if kind == 0:                                              # heavy ties
        s = rng.integers(0, 6, n) / 6.0
    elif kind == 1:
        s = rng.random(n)
    else:                                                      # mostly increasing with a few dips
        s = np.sort(rng.random(n)) - 0.4 * (rng.random(n) < 0.3) * rng.random(n)
    tm = rng.choice([0.02, 0.05, 0.1, 0.25, 0.5], n) * (0.5 + rng.random(n))
    tp = tm if rng.random() < 0.3 else rng.choice([0.02, 0.05, 0.1, 0.25, 0.5], n) * (0.5 + rng.random(n))
    if ZERO_SIDES:                                             # one-sided zero tolerances (support endpoint = modal value)
        tm = np.where(rng.random(n) < 0.25, 0.0, tm); tp = np.where(rng.random(n) < 0.25, 0.0, tp)
    return s, tm, tp


def random_cutoff(s, tm, tp, gamma):
    l, u = band_ends(s, tm, tp, gamma)
    pool = np.concatenate([s, l, u, rng.random(4) * (s.max() - s.min() + 0.2) + s.min() - 0.1])
    return float(rng.choice(pool))


def bisect(pred, hi=1e7, iters=140):
    """smallest gamma>=0 with pred(gamma) True (pred monotone: False ... True)."""
    if pred(0.0):
        return 0.0
    lo = 0.0
    while not pred(hi):
        hi *= 10
        if hi > 1e30:
            return INF
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if pred(mid):
            hi = mid
        else:
            lo = mid
    return hi

INF = float("inf")


def suite(zero_sides: bool):
    global ZERO_SIDES
    ZERO_SIDES = zero_sides
    res = {"zero_sided_tolerances": zero_sides}
    log(f"--- suite: {'with one-sided zero tolerances' if zero_sides else 'strictly positive tolerances'} ---")

    # ---------------- V1: band characterisation vs corner enumeration
    mism1 = 0; cnt1 = 0; n_in_Z = n_in_P = n_notP = 0
    for _ in range(12000):
        n = int(rng.integers(2, 9))
        s, tm, tp = random_instance(n)
        gamma = float(rng.choice([0.0, 0.3, 0.7, 1.0, 1.5, 3.0]) * (0.3 + rng.random()))
        Z = certain_invalid_bands(s, tm, tp, gamma); P = possible_invalid_bands(s, tm, tp, gamma)
        for _c in range(3):
            c = random_cutoff(s, tm, tp, gamma)
            ex_valid, ex_invalid = corner_oracle(c, s, tm, tp, gamma)
            inZ, inP = in_bands(c, Z), in_bands(c, P)
            cnt1 += 1
            n_in_Z += inZ; n_in_P += inP; n_notP += (not inP)
            if inZ != (not ex_valid) or inP != ex_invalid:
                mism1 += 1
    res["V1"] = {"cutoff_checks": cnt1, "mismatches": mism1, "in_Z": n_in_Z, "in_P": n_in_P, "robustly_valid": n_notP}
    log(f"V1 band characterisation vs corner enumeration: {cnt1} checks, mismatches {mism1} "
        f"(in Z {n_in_Z}, in P {n_in_P}, outside P {n_notP})")

    # ---------------- V2: radii vs bisection on the corner oracle
    maxdiff_rho = maxdiff_phi = 0.0; cnt2 = 0
    for _ in range(1500):
        n = int(rng.integers(2, 8))
        s, tm, tp = random_instance(n)
        c = random_cutoff(s, tm, tp, 1.0)
        f_rho = bisect(lambda g: corner_oracle(c, s, tm, tp, g)[0])
        f_phi = bisect(lambda g: corner_oracle(c, s, tm, tp, g)[1])
        r, p = rho(c, s, tm, tp), phi(c, s, tm, tp)
        if np.isfinite(r) or np.isfinite(f_rho):
            maxdiff_rho = max(maxdiff_rho, abs(r - f_rho) / max(1.0, abs(r)) if np.isfinite(r) and np.isfinite(f_rho) else INF)
        if np.isfinite(p) or np.isfinite(f_phi):
            maxdiff_phi = max(maxdiff_phi, abs(p - f_phi) / max(1.0, abs(p)) if np.isfinite(p) and np.isfinite(f_phi) else INF)
        cnt2 += 1
    res["V2"] = {"instances": cnt2, "max_rel_diff_rho": maxdiff_rho, "max_rel_diff_phi": maxdiff_phi}
    log(f"V2 rho/phi vs bisection on corner oracle: {cnt2} instances, max rel. diff rho {maxdiff_rho:.2e}, phi {maxdiff_phi:.2e}")

    # ---------------- V3: global radius vs LP, rho(c*) = R, max_c rho = R
    maxdiff_lp = 0.0; viol_cstar = 0; viol_sup = 0; cnt3 = 0; n_inv = 0; n_infinite = 0
    for _ in range(3000):
        n = int(rng.integers(2, 13))
        s, tm, tp = random_instance(n)
        R, pair = global_radius(s, tm, tp)
        if not np.isfinite(R):                                  # unrepairable inversion: LP infeasible
            n_infinite += 1
            continue
        # LP: min r  s.t.  s_q - r*tm_q <= y_q <= s_q + r*tp_q, y_q <= y_{q+1}
        nv = n + 1
        cost = np.zeros(nv); cost[-1] = 1.0
        A, b = [], []
        for q in range(n):
            row = np.zeros(nv); row[q] = -1.0; row[-1] = -tm[q]; A.append(row); b.append(-s[q])   # s_q - r tm <= y
            row = np.zeros(nv); row[q] = 1.0; row[-1] = -tp[q]; A.append(row); b.append(s[q])       # y <= s_q + r tp
        for q in range(n - 1):
            row = np.zeros(nv); row[q] = 1.0; row[q + 1] = -1.0; A.append(row); b.append(0.0)
        sol = linprog(cost, A_ub=np.array(A), b_ub=np.array(b), bounds=[(None, None)] * n + [(0, None)], method="highs")
        maxdiff_lp = max(maxdiff_lp, abs(sol.fun - R))
        cnt3 += 1
        if pair is not None:
            n_inv += 1
            cs, Rc = critical_cutoff(s, tm, tp, pair)
            # attained at c* for strictly positive tolerances; with a zero side it is a supremum (approached from one side)
            val = max(rho(cs, s, tm, tp), rho(cs - 1e-9, s, tm, tp), rho(cs + 1e-9, s, tm, tp))
            exact_ok = abs(rho(cs, s, tm, tp) - R) <= 1e-9 * max(1.0, R)
            if (not zero_sides and not exact_ok) or val < R - 1e-6 * max(1.0, R):
                viol_cstar += 1
        # sup of rho on a grid never exceeds R
        grid = np.linspace(s.min() - 0.1, s.max() + 0.1, 401)
        if max(rho(c, s, tm, tp) for c in grid) > R + 1e-9 * max(1.0, R):
            viol_sup += 1
    res["V3"] = {"instances": cnt3, "with_inversion": n_inv, "max_abs_diff_LP": maxdiff_lp,
                 "R_infinite_skipped": n_infinite,
                 "rho_at_critical_cutoff_violations": viol_cstar, "rho_above_R_violations": viol_sup}
    log(f"V3 R vs LP: {cnt3} instances ({n_inv} with an inversion, {n_infinite} with R=inf skipped), max |R-LP| {maxdiff_lp:.2e}; "
        f"rho(c*)=R violations {viol_cstar}; rho>R violations {viol_sup}")

    # ---------------- V4: nesting, gamma=0 equals Theorem 1, emptiness threshold
    bad_sub = bad_nest = bad_spec = bad_empty = 0; cnt4 = 0
    for _ in range(6000):
        n = int(rng.integers(2, 12))
        s, tm, tp = random_instance(n)
        g1, g2 = sorted(rng.random(2) * 2.0)
        Z1, Z2 = certain_invalid_bands(s, tm, tp, g1), certain_invalid_bands(s, tm, tp, g2)
        P1, P2 = possible_invalid_bands(s, tm, tp, g1), possible_invalid_bands(s, tm, tp, g2)
        for _c in range(6):
            c = random_cutoff(s, tm, tp, g2)
            cnt4 += 1
            if in_bands(c, Z1) and not in_bands(c, P1): bad_sub += 1
            if in_bands(c, Z2) and not in_bands(c, Z1): bad_nest += 1        # Z shrinks as gamma grows
            if in_bands(c, P1) and not in_bands(c, P2): bad_nest += 1        # P grows
            acc = s >= c
            direct_invalid = not is_suffix(acc)
            if in_bands(c, certain_invalid_bands(s, tm, tp, 0.0)) != direct_invalid: bad_spec += 1
            if in_bands(c, possible_invalid_bands(s, tm, tp, 0.0)) != direct_invalid: bad_spec += 1
        R, _p = global_radius(s, tm, tp)
        if not np.isfinite(R):
            if len(certain_invalid_bands(s, tm, tp, 1e6)) == 0: bad_empty += 1
        elif R > 0:
            if len(certain_invalid_bands(s, tm, tp, R * (1 + 1e-9))) != 0: bad_empty += 1
            if len(certain_invalid_bands(s, tm, tp, R * (1 - 1e-6))) == 0: bad_empty += 1
        else:
            if len(certain_invalid_bands(s, tm, tp, 0.0)) != 0: bad_empty += 1
    res["V4"] = {"cutoff_checks": cnt4, "Z_subset_P_violations": bad_sub, "nesting_violations": bad_nest,
                 "gamma0_vs_Theorem1_mismatches": bad_spec, "emptiness_threshold_violations": bad_empty}
    log(f"V4 structure: {cnt4} cutoff checks; Z<=P violations {bad_sub}; nesting {bad_nest}; "
        f"gamma=0 vs Theorem 1 {bad_spec}; Z empty iff gamma>=R violations {bad_empty}")

    # ---------------- V5: exact maximum-margin cutoff vs dense grid (strictly positive tolerances only)
    if zero_sides:
        return res
    worse = 0; cnt5 = 0; maxgap = 0.0
    for _ in range(400):
        n = int(rng.integers(3, 10))
        s, tm, tp = random_instance(n)
        lo = float(s.min() + 0.15 * (s.max() - s.min())); hi = float(lo + (0.1 + 0.4 * rng.random()) * (s.max() - s.min() + 1e-9))
        c_best, m_best = max_margin_cutoff(s, tm, tp, lo, hi)
        grid = np.linspace(lo, hi, 4001)
        mg = max(margin(c, s, tm, tp) for c in grid)
        cnt5 += 1
        if m_best < mg - 1e-12: worse += 1
        maxgap = max(maxgap, m_best - mg)
    res["V5"] = {"windows": cnt5, "exact_below_grid_violations": worse, "max_exact_minus_grid": maxgap}
    log(f"V5 maximum-margin cutoff: {cnt5} windows, exact < grid violations {worse}, max (exact - grid) {maxgap:.2e}")

    return res


def main():
    t0 = time.time()
    res = {"seed": 20261020, "positive_tolerances": suite(False), "zero_sided_tolerances": suite(True)}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "margin_verification_summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (LOGD / "margin_verification.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(json.dumps(res))


if __name__ == "__main__":
    main()
