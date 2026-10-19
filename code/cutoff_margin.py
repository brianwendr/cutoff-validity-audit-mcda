#!/usr/bin/env python3
"""
Cutoff-level tolerance quantities for a scalar score on a finite chain (core library, numpy only).

Notation (manuscript Section 5).  Scores s_0..s_N, lower/upper tolerances tm_q, tp_q > 0
(symmetric case tm = tp = tau).  At tolerance level gamma >= 0 the admissible band of alternative q is
    [l_q, u_q] = [s_q - gamma*tm_q, s_q + gamma*tp_q]
(the alpha-cut with alpha = 1 - gamma when (s_q - tm_q, s_q, s_q + tp_q) is a triangular fuzzy score).

    Z_gamma = U_{i<j} (u_j, l_i]    cutoffs that are invalid for EVERY admissible score vector
    P_gamma = U_{i<j} (l_j, u_i]    cutoffs that are invalid for SOME admissible score vector
    rho(c)  = max_{i<j} min{(s_i-c)_+/tm_i , (c-s_j)_+/tp_j}      repair radius of the cutoff c
    phi(c)  = min_{i<j} max{(c-s_i)_+/tp_i , (s_j-c)_+/tm_j}      fragility (break radius) of the cutoff c
    m(c)    = phi(c) - rho(c)                                       signed tolerance margin
    R       = max_{i<j} (s_i-s_j)_+/(tm_i+tp_j) = max_c rho(c)      global monotone-repair radius

All functions are O(N) per cutoff (prefix maxima/minima) except where stated.
"""
from __future__ import annotations
import numpy as np

INF = float("inf")


def _down(s, c, t):
    """Cost of pushing a score at s strictly below c: (s-c)_+/t.  With a zero lower tolerance a score s >= c can
    never be rejected, so the cost is +inf (this includes the tie s = c); a score below c costs 0."""
    s = np.asarray(s, float); t = np.asarray(t, float)
    x = np.maximum(s - c, 0.0)
    out = np.where(t > 0, x / np.where(t > 0, t, 1.0), 0.0)
    return np.where((t <= 0) & (s >= c), INF, out)


def _div(x, t):
    """x_+ / t with the conventions x/0 = +inf for x > 0 and 0/0 = 0 (one-sided zero tolerances are allowed)."""
    x = np.maximum(np.asarray(x, float), 0.0)
    t = np.asarray(t, float)
    out = np.zeros_like(x)
    pos = t > 0
    out[pos] = x[pos] / t[pos] if np.ndim(t) else x[pos] / t
    out[~pos & (x > 0)] = INF
    return out


# ----------------------------------------------------------------------------- bands
def band_ends(s, tm, tp, gamma):
    s = np.asarray(s, float)
    return s - gamma * np.asarray(tm, float), s + gamma * np.asarray(tp, float)


def merge_halfopen(iv):
    """Merge intervals (a, b] ; touching intervals (a1,b1],(b1,b2] merge."""
    iv = sorted((a, b) for a, b in iv if b > a)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def certain_invalid_bands(s, tm, tp, gamma):
    """Z_gamma = U_j (u_j, K_j],  K_j = max_{i<j} l_i."""
    l, u = band_ends(s, tm, tp, gamma)
    iv, K = [], -INF
    for j in range(len(l)):
        if j > 0 and K > u[j]:
            iv.append((u[j], K))
        K = max(K, l[j])
    return merge_halfopen(iv)


def possible_invalid_bands(s, tm, tp, gamma):
    """P_gamma = U_j (l_j, Mu_j],  Mu_j = max_{i<j} u_i."""
    l, u = band_ends(s, tm, tp, gamma)
    iv, M = [], -INF
    for j in range(len(l)):
        if j > 0 and M > l[j]:
            iv.append((l[j], M))
        M = max(M, u[j])
    return merge_halfopen(iv)


def in_bands(c, bands):
    return any(a < c <= b for a, b in bands)


def band_length(bands, lo=-INF, hi=INF):
    tot = 0.0
    for a, b in bands:
        a2, b2 = max(a, lo), min(b, hi)
        if b2 > a2:
            tot += b2 - a2
    return tot


def covered(lo, hi, bands):
    """True iff every point of the closed window [lo, hi] lies in the union of the half-open bands (a, b]."""
    for a, b in bands:
        if a < lo and hi <= b:
            return True
    return False


# ----------------------------------------------------------------------------- cutoff-level radii
def rho(c, s, tm, tp):
    """Repair radius: smallest gamma at which cutoff c can be made valid by an admissible score vector (infimum)."""
    s = np.asarray(s, float)
    a = _down(s, c, tm)    # must be pushed down to be rejected
    b = _div(c - s, tp)    # must be pushed up to be accepted
    best, pm = 0.0, -INF
    for j in range(len(s)):
        if j > 0:
            best = max(best, min(pm, b[j]))
        pm = max(pm, a[j])
    return best


def phi(c, s, tm, tp):
    """Fragility: smallest gamma at which cutoff c can become invalid (infimum)."""
    s = np.asarray(s, float)
    a = _div(c - s, tp)    # i accepted: push up to reach c
    b = _down(s, c, tm)    # j rejected: push down below c
    best, pmin = INF, INF
    for j in range(len(s)):
        if j > 0:
            best = min(best, max(pmin, b[j]))
        pmin = min(pmin, a[j])
    return best


def margin(c, s, tm, tp):
    """Signed margin: -rho(c) if c is (strictly) inside an inversion band, else phi(c).
    (rho(c) > 0 forces phi(c) = 0, and phi(c) > 0 forces rho(c) = 0, so this equals phi - rho whenever finite.)"""
    r = rho(c, s, tm, tp)
    return -r if r > 0 else phi(c, s, tm, tp)


def global_radius(s, tm, tp):
    """R = max_{i<j} (s_i - s_j)_+ / (tm_i + tp_j) and the maximising pair."""
    s = np.asarray(s, float); tm = np.asarray(tm, float); tp = np.asarray(tp, float)
    best, pair = 0.0, None
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            d = max(s[i] - s[j], 0.0)
            v = 0.0 if d == 0 else (d / (tm[i] + tp[j]) if tm[i] + tp[j] > 0 else INF)
            if v > best:
                best, pair = v, (i, j)
    return best, pair


def critical_cutoff(s, tm, tp, pair):
    i, j = pair
    R = (s[i] - s[j]) / (tm[i] + tp[j])
    return s[j] + tp[j] * R, R


# ----------------------------------------------------------------------------- exact maximum-margin cutoff
def _lines(s, tm, tp):
    L = []
    for q in range(len(s)):
        L.append((-1.0 / tm[q], s[q] / tm[q]))      # A_q(c) = (s_q - c)/tm_q   (tolerances must be > 0 here)
        L.append((1.0 / tp[q], -s[q] / tp[q]))      # B_q(c) = (c - s_q)/tp_q
    return L


def candidate_cutoffs(s, tm, tp, lo, hi):
    """Window endpoints plus all pairwise intersections of the 2N+2 lines that define rho and phi
    (including their intersections with 0, i.e. the scores themselves).  The margin is piecewise linear
    with kinks only at such points, so its maximum over [lo, hi] is attained here."""
    s = np.asarray(s, float); tm = np.asarray(tm, float); tp = np.asarray(tp, float)
    L = _lines(s, tm, tp)
    cands = {lo, hi}
    for q in range(len(s)):
        cands.add(float(s[q]))
    for x in range(len(L)):
        for y in range(x + 1, len(L)):
            k1, d1 = L[x]; k2, d2 = L[y]
            if k1 != k2:
                cands.add((d2 - d1) / (k1 - k2))
    return sorted(c for c in cands if lo <= c <= hi)


def max_margin_cutoff(s, tm, tp, lo, hi):
    best_c, best_m = None, -INF
    for c in candidate_cutoffs(s, tm, tp, lo, hi):
        m = margin(c, s, tm, tp)
        if m > best_m + 1e-15:
            best_c, best_m = c, m
    return best_c, best_m
