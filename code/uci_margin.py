#!/usr/bin/env python3
"""
Applies the cutoff-level quantities of Section 5 to the UCI geometry-profile illustration (numpy only).
Reads outputs/norm_specific_open_data_scores_wide.csv (p = 1, 2, infinity TOPSIS closeness of the 12 profiles).
Tolerances: symmetric  tau_q = (b_q - a_q)/2  (as in Section 9.2) and support-faithful asymmetric
(tm_q, tp_q) = (s_q - a_q, b_q - s_q), where [a_q, b_q] is the min/max over the three norms and s_q the score of the
norm being audited.  Writes outputs/uci_margin_*.csv and outputs/uci_margin_summary.json.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from cutoff_margin import (certain_invalid_bands, possible_invalid_bands, in_bands, band_length, rho, phi, margin,
                           global_radius, critical_cutoff, max_margin_cutoff, covered)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
C0 = 0.55
WINDOW = (0.50, 0.60)
NORMS = ["p1", "p2", "pinf"]


def load():
    rows = list(csv.DictReader((OUT / "norm_specific_open_data_scores_wide.csv").open()))
    S = {k: np.array([float(r[k]) for r in rows]) for k in NORMS}
    a = np.minimum.reduce([S[k] for k in NORMS]); b = np.maximum.reduce([S[k] for k in NORMS])
    return S, a, b


def fmt_bands(B):
    return "; ".join(f"({x:.4f},{y:.4f}]" for x, y in B) if B else "empty"


def main():
    S, a, b = load()
    tau = (b - a) / 2.0
    summary = {"c0": C0, "window": list(WINDOW), "tau_symmetric": [round(float(x), 10) for x in tau]}
    rows = []
    for k in NORMS:
        s = S[k]
        rng_ = float(s.max() - s.min())
        for mode in ("symmetric", "asymmetric"):
            tm, tp = (tau, tau) if mode == "symmetric" else (s - a, b - s)
            R, pair = global_radius(s, tm, tp)
            rec = {"norm": k, "tolerance": mode, "R": float(R), "worst_pair": list(pair) if pair else None}
            if np.isfinite(R) and mode == "symmetric":
                cs, _ = critical_cutoff(s, tm, tp, pair)
                rec["critical_cutoff"] = float(cs)
            if mode == "asymmetric" and pair is not None and np.isfinite(R):
                i, j = pair
                rec["critical_cutoff"] = float(s[j] + tp[j] * R) if tp[j] > 0 else float(s[i] - tm[i] * R)
            for g in (0.5, 1.0, 2.0):
                Z = certain_invalid_bands(s, tm, tp, g); P = possible_invalid_bands(s, tm, tp, g)
                lz = band_length(Z, s.min(), s.max()); lp = band_length(P, s.min(), s.max())
                rec[f"gamma_{g}"] = {"Z": fmt_bands(Z), "P": fmt_bands(P),
                                      "Z_share_of_range": lz / rng_, "P_minus_Z_share": (lp - lz) / rng_,
                                      "robustly_valid_share": 1 - lp / rng_,
                                      "c0_in_Z": bool(in_bands(C0, Z)), "c0_in_P": bool(in_bands(C0, P)),
                                      "window_has_robustly_valid_cutoff": not covered(WINDOW[0], WINDOW[1], P)}
            r0, p0 = rho(C0, s, tm, tp), phi(C0, s, tm, tp)
            rec["c0_rho"] = float(r0); rec["c0_phi"] = float(p0); rec["c0_margin"] = float(margin(C0, s, tm, tp))
            if mode == "symmetric":                               # exact maximum-margin cutoff needs strictly positive tolerances
                cb, mb = max_margin_cutoff(s, tm, tp, WINDOW[0], WINDOW[1])
                rec["window_best_cutoff"] = float(cb); rec["window_best_margin"] = float(mb)
                # also the best cutoff over the whole score range
                cb2, mb2 = max_margin_cutoff(s, tm, tp, float(s.min()), float(s.max()))
                rec["range_best_cutoff"] = float(cb2); rec["range_best_margin"] = float(mb2)
            summary.setdefault(k, {})[mode] = rec
            rows.append((k, mode, rec))
    # margin-optimal cutoff when the policy fixes how many alternatives may be accepted (quota window [kmin, kmax])
    quota_rows = []
    for k in ("p1", "p2"):
        s = S[k]; srt = np.sort(s)[::-1]
        for kmin, kmax in ((1, 3), (2, 4), (3, 5)):
            lo, hi = float(srt[kmax - 1]), float(srt[kmin - 1])
            c, m = max_margin_cutoff(s, tau, tau, lo, hi)
            acc = [int(q) for q in range(len(s)) if s[q] >= c]
            flags = np.array([s[q] >= c for q in range(len(s))])
            nominal_valid = not bool(np.any(flags[:-1] & ~flags[1:]))
            quota_rows.append({"norm": k, "kmin": kmin, "kmax": kmax, "window_lo": round(lo, 6), "window_hi": round(hi, 6),
                               "best_cutoff": round(float(c), 6), "margin": round(float(m), 6), "accepted": " ".join(map(str, acc)),
                               "nominally_valid": nominal_valid, "rho": round(float(rho(c, s, tau, tau)), 6), "phi": round(float(phi(c, s, tau, tau)), 6)})
    with (OUT / "uci_margin_quota.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(quota_rows[0].keys())); w.writeheader(); w.writerows(quota_rows)
    summary["quota"] = quota_rows
    for r in quota_rows:
        print("quota", r)
    # profile over a grid, symmetric tolerances
    grid = np.linspace(0.25, 0.75, 501)
    with (OUT / "uci_margin_profile.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["c"] + [f"{k}_{x}" for k in NORMS for x in ("rho", "phi", "margin")])
        for c in grid:
            row = [round(float(c), 6)]
            for k in NORMS:
                s = S[k]
                row += [round(float(rho(c, s, tau, tau)), 6), round(float(phi(c, s, tau, tau)), 6),
                        round(float(margin(c, s, tau, tau)), 6)]
            w.writerow(row)
    # phase-diagram bands: Z_gamma and P_gamma over a gamma grid (symmetric, p=2 and p=1)
    with (OUT / "uci_margin_phase.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["norm", "gamma", "kind", "lo_open", "hi_closed"])
        for k in NORMS:
            for g in np.round(np.linspace(0.0, 7.0, 141), 4):
                for kind, fn in (("Z", certain_invalid_bands), ("P", possible_invalid_bands)):
                    for lo, hi in fn(S[k], tau, tau, float(g)):
                        w.writerow([k, g, kind, round(float(lo), 6), round(float(hi), 6)])
    (OUT / "uci_margin_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for k in NORMS:
        for mode in ("symmetric", "asymmetric"):
            r = summary[k][mode]
            print(k, mode, "R=%.4f" % r["R"], r["worst_pair"], r.get("critical_cutoff"),
                  "| c0 rho=%.4f phi=%.4f m=%.4f" % (r["c0_rho"], r["c0_phi"], r["c0_margin"]))
            for g in (0.5, 1.0, 2.0):
                x = r[f"gamma_{g}"]
                print("   gamma", g, "Z:", x["Z"], "| P:", x["P"], "| Zshare %.3f Pshare-Z %.3f valid %.3f" % (
                    x["Z_share_of_range"], x["P_minus_Z_share"], x["robustly_valid_share"]), "c0 in Z/P:", x["c0_in_Z"], x["c0_in_P"],
                    "win robust valid:", x["window_has_robustly_valid_cutoff"])
            if mode == "symmetric":
                print("   best in window", r["window_best_cutoff"], r["window_best_margin"], "| best over range", r["range_best_cutoff"], r["range_best_margin"])


if __name__ == "__main__":
    main()
