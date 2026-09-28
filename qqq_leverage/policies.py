"""Leverage policies: each returns the target leverage array (n, N) used while the
regime is ON (the regime filter still sends everything to SGOV when OFF).
All inputs are known at the close of day t."""
import numpy as np
import pandas as pd


def static(F, L):
    return np.full(F["P"].shape, float(L))


def _dd(F, anchor):
    if anchor == "ath":
        return F["dd"]
    if anchor == "52w":
        return F["dd52"]
    if anchor == "regime":          # peak since the regime last switched ON
        return F["dd_reg"]
    raise ValueError(anchor)


def add_regime_drawdown(F, reg):
    """Drawdown measured from the highest close since the regime last turned ON."""
    P = F["P"]
    n, N = P.shape
    out = np.zeros((n, N))
    peak = P[:, 0].copy()
    prev = reg[:, 0].copy()
    for t in range(N):
        entered = (reg[:, t] == 1) & (prev == 0)
        peak = np.where(entered, P[:, t], np.maximum(peak, P[:, t]))
        out[:, t] = P[:, t] / peak - 1
        prev = reg[:, t]
    F["dd_reg"] = out
    return F


def ladder(F, base, steps, anchor="ath", sticky=True, reset=0.0):
    """steps = [(drawdown, leverage), ...] e.g. [(0.05, 1.4), (0.10, 1.5)].
    sticky: once raised, hold until the index is back within `reset` of its high.
    non-sticky: leverage follows the current drawdown level."""
    dd = _dd(F, anchor)
    n, N = dd.shape
    lvl = np.full((n, N), float(base))
    for thr, lev in sorted(steps):
        lvl = np.where(dd <= -thr, lev, lvl)
    if not sticky:
        return lvl
    out = np.empty((n, N))
    cur = np.full(n, float(base))
    for t in range(N):
        cur = np.maximum(cur, lvl[:, t])
        cur = np.where(dd[:, t] >= -reset, lvl[:, t], cur)
        out[:, t] = cur
    return out


def voltarget(F, target, hl=20, lmin=1.0, lmax=2.0):
    return np.clip(target / F[f"vol{hl}"], lmin, lmax)


def invvar(F, lref, vref, hl=20, lmin=1.0, lmax=2.0):
    """Kelly-style: leverage ∝ 1/σ²  (lref at vol = vref)."""
    return np.clip(lref * (vref / F[f"vol{hl}"]) ** 2, lmin, lmax)


def sma_dist(F, l_near, l_far, d_far=0.20):
    """Linear in distance above the 200-day SMA: l_near at 0%, l_far at d_far."""
    x = np.clip(F["dist"] / d_far, 0, 1)
    return l_near + (l_far - l_near) * x


def oversold(F, base, boost, rsi_thr=10, hold=5):
    rsi = F["rsi2"]
    n, N = rsi.shape
    out = np.empty((n, N))
    cnt = np.zeros(n)
    for t in range(N):
        cnt = np.where(rsi[:, t] < rsi_thr, hold, np.maximum(cnt - 1, 0))
        out[:, t] = np.where(cnt > 0, base + boost, base)
    return out


def entry_boost(F, reg, base, boost_lev, days):
    """Higher leverage for `days` after each regime re-entry."""
    n, N = reg.shape
    out = np.empty((n, N))
    cnt = np.zeros(n)
    prev = reg[:, 0].copy()
    for t in range(N):
        entered = (reg[:, t] == 1) & (prev == 0)
        cnt = np.where(entered, days, np.maximum(cnt - 1, 0))
        out[:, t] = np.where(cnt > 0, boost_lev, base)
        prev = reg[:, t]
    return out


def vol_gate(F, Lhi, Llo, vcut, hl=20):
    """Step function: Lhi when vol < vcut else Llo."""
    return np.where(F[f"vol{hl}"] < vcut, Lhi, Llo)


def vt_ladder(F, target, steps, hl=20, lmin=1.0, lmax=2.0, anchor="ath"):
    """Vol target, plus extra leverage on drawdowns (additive), capped."""
    base = voltarget(F, target, hl, lmin, lmax)
    add = ladder(F, 0.0, steps, anchor=anchor, sticky=False)
    return np.clip(base + add, lmin, lmax)


def dip_calm(F, base, steps, vmax, hl=20, anchor="ath"):
    """Dip ladder that only applies while realised vol is below vmax."""
    lad = ladder(F, base, steps, anchor=anchor, sticky=False)
    return np.where(F[f"vol{hl}"] < vmax, lad, base)


def sma_steps(F, levels, cuts=(0.04, 0.08, 0.12, 0.16), below=None):
    """Step leverage by distance above the 200-SMA.
    levels[k] applies for dist in [cuts[k-1], cuts[k]); len(levels) == len(cuts) + 1.
    below: optional separate leverage when price is under the SMA (still in regime)."""
    dist = F["dist"]
    out = np.full(dist.shape, float(levels[-1]))
    for k in range(len(cuts) - 1, -1, -1):
        out = np.where(dist < cuts[k], levels[k], out)
    if below is not None:
        out = np.where(dist < 0, below, out)
    return out


def sma_dist_os(F, l_near, l_far, d_far, boost, rsi_thr=5, hold=5, lmax=2.0):
    """SMA-distance leverage plus a short-term oversold (RSI2) boost."""
    base = sma_dist(F, l_near, l_far, d_far)
    os_ = oversold(F, 0.0, boost, rsi_thr, hold)
    return np.minimum(base + os_, lmax)


def vt_os(F, target, hl, lmin, lmax, boost, rsi_thr=5, hold=5, cap=2.0):
    """Vol target plus a short-term oversold (RSI2) boost."""
    return np.minimum(voltarget(F, target, hl, lmin, lmax) + oversold(F, 0.0, boost, rsi_thr, hold), cap)


def sma_volcap(F, levels, target, hl=20, lmin=1.0):
    """SMA-distance steps, but never above the vol-target leverage."""
    return np.maximum(np.minimum(sma_steps(F, levels), target / F[f"vol{hl}"]), lmin)


def blend(*arrs):
    return np.mean(arrs, axis=0)


def turn_of_month(dates_like, shape, base, boost, before=1, after=3):
    """Calendar rule (history only): boost on the last `before` and first `after`
    trading days of each month."""
    d = pd.DatetimeIndex(dates_like)
    m = d.to_period("M")
    k_from_start = pd.Series(1, index=d).groupby(m).cumcount().values
    k_to_end = pd.Series(1, index=d)[::-1].groupby(m[::-1]).cumcount().values[::-1]
    earn = (k_from_start < after) | (k_to_end < before)     # days that should carry the boost
    decide = np.r_[earn[1:], False]                          # set at the prior close
    return np.broadcast_to(np.where(decide, base + boost, base), shape).astype(float)


def drop_boost(F, base, boost, days=3, drop=0.05, hold=5):
    """Boost after a sharp short-term selloff: `days`-day return <= -drop."""
    P = F["P"]
    ret = np.full(P.shape, np.nan)
    ret[:, days:] = P[:, days:] / P[:, :-days] - 1
    n, N = P.shape
    out = np.empty((n, N))
    cnt = np.zeros(n)
    trig = np.nan_to_num(ret, nan=0.0) <= -drop
    for t in range(N):
        cnt = np.where(trig[:, t], hold, np.maximum(cnt - 1, 0))
        out[:, t] = np.where(cnt > 0, base + boost, base)
    return out


def deep_dip(F, base_arr, steps, anchor="ath"):
    """Add leverage only while the index is at least X below its high (level-based,
    so it comes off again as soon as the index recovers above the trigger).
    base_arr may be a scalar or an array (e.g. a vol-target leverage)."""
    add = ladder(F, 0.0, [(thr, a) for thr, a in steps], anchor=anchor, sticky=False)
    return base_arr + add
