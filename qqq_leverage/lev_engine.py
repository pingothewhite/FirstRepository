"""Vectorised engine for QQQ regime-filtered leverage policies.

Replicates the v5 dashboard's mechanics:
  * 200-day SMA band regime (band_up / band_dn, entry_days / exit_days, counters
    persist while price is inside the band, exactly as build_regime_band()).
  * In regime: exposure L built from QLD (L-1) + QQQ (2-L); daily cost
    (L-1)*QLD_DRAG + (2-L)*QQQ_DRAG.  Out of regime: SGOV yield.
  * Price-return (Close, not dividend-adjusted), like the dashboard.

Arrays are (n_paths, n_days); a historical backtest is n_paths = 1.
Leverage decided at the close of day t is applied to the return of day t+1.
"""
import numpy as np
import pandas as pd

QLD_DRAG_A = 0.0545      # cost per unit of leverage above 1x (QLD sleeve)
QQQ_DRAG_A = 0.002
SGOV_A = 0.042
TC = 0.0005              # 5 bp per unit of exposure traded (spread + slippage)

BAND_UP, BAND_DN, ENTRY, EXIT = 0.04, 0.08, 3, 15


def load_close(path):
    d = pd.read_excel(path)
    d["Date"] = pd.to_datetime(d["Date"]).dt.normalize()
    return d.set_index("Date")["Close"].astype(float).sort_index()


# ------------------------------------------------------------------ features
def _rolling_mean(P, w):
    cs = np.cumsum(P, axis=1)
    out = np.full(P.shape, np.nan)
    out[:, w - 1:] = cs[:, w - 1:]
    out[:, w:] -= cs[:, :-w]
    out[:, w - 1:] /= w
    return out


def _rolling_max(P, w):
    return pd.DataFrame(P.T).rolling(w, min_periods=1).max().values.T


def features(P, vol_hls=(10, 20, 40)):
    n, N = P.shape
    r = np.zeros_like(P)
    r[:, 1:] = P[:, 1:] / P[:, :-1] - 1
    F = {"P": P, "r": r}
    F["sma"] = _rolling_mean(P, 200)
    F["dist"] = P / F["sma"] - 1
    F["ath"] = np.maximum.accumulate(P, axis=1)
    F["dd"] = P / F["ath"] - 1
    F["hi252"] = _rolling_max(P, 252)
    F["dd52"] = P / F["hi252"] - 1
    r2 = r ** 2
    v0 = np.nanmean(r2[:, 1:22], axis=1)
    for hl in vol_hls:
        lam = 0.5 ** (1 / hl)
        v = np.empty_like(P)
        cur = v0.copy()
        for t in range(N):
            cur = lam * cur + (1 - lam) * r2[:, t]
            v[:, t] = cur
        F[f"vol{hl}"] = np.sqrt(252 * v)
    # RSI(2), Wilder
    up = np.clip(r, 0, None)
    dn = np.clip(-r, 0, None)
    a = 0.5
    u = np.zeros(n)
    d = np.zeros(n)
    rsi = np.empty_like(P)
    for t in range(N):
        u = (1 - a) * u + a * up[:, t]
        d = (1 - a) * d + a * dn[:, t]
        rsi[:, t] = 100 * u / np.maximum(u + d, 1e-12)
    F["rsi2"] = rsi
    F["mom252"] = np.full_like(P, np.nan)
    F["mom252"][:, 252:] = P[:, 252:] / P[:, :-252] - 1
    return F


def regime(F, start, init=None, band_up=BAND_UP, band_dn=BAND_DN, entry=ENTRY, exit_=EXIT):
    """init = (state, above_cnt, below_cnt) arrays/scalars, else derived at `start`."""
    P, sma = F["P"], F["sma"]
    n, N = P.shape
    reg = np.zeros((n, N), np.int8)
    upper, lower = sma * (1 + band_up), sma * (1 - band_dn)
    if init is None:
        state = (P[:, start] > sma[:, start]).astype(np.int8)
        a = np.zeros(n, int)
        b = np.zeros(n, int)
    else:
        state = np.broadcast_to(np.asarray(init[0], np.int8), (n,)).copy()
        a = np.broadcast_to(np.asarray(init[1]), (n,)).astype(int).copy()
        b = np.broadcast_to(np.asarray(init[2]), (n,)).astype(int).copy()
    for i in range(start, N):
        up = P[:, i] > upper[:, i]
        dn = P[:, i] < lower[:, i]
        a = np.where(up, a + 1, np.where(dn, 0, a))
        b = np.where(dn, b + 1, np.where(up, 0, b))
        go_in = (state == 0) & (a >= entry)
        go_out = (state == 1) & (b >= exit_)
        state = np.where(go_in, 1, np.where(go_out, 0, state)).astype(np.int8)
        rs = go_in | go_out
        a = np.where(rs, 0, a)
        b = np.where(rs, 0, b)
        reg[:, i] = state
    return reg, (state, a, b)


# ------------------------------------------------------------------ execution
def execute(Ltgt, reg, start, band=0.0):
    """Exposure after each close. Entries/exits always trade; in-regime leverage
    changes only when |target - current| >= band (no-trade band)."""
    n, N = reg.shape
    X = np.zeros((n, N))
    cur = np.zeros(n)
    Lt = np.broadcast_to(Ltgt, (n, N)) if np.ndim(Ltgt) else np.full((n, N), float(Ltgt))
    for t in range(start, N):
        tgt = np.where(reg[:, t] == 1, Lt[:, t], 0.0)
        chg = (np.abs(tgt - cur) >= band - 1e-12) | (tgt == 0) | (cur == 0)
        cur = np.where(chg, tgt, cur)
        X[:, t] = cur
    return X


def daily_factor(X, r, start, qld=QLD_DRAG_A, qqq=QQQ_DRAG_A, sgov=SGOV_A, tc=TC):
    """g[:, t+1] = growth factor of day t+1 using exposure X[:, t]."""
    qd, qq, sg = qld / 252, qqq / 252, sgov / 252
    n, N = X.shape
    g = np.ones((n, N))
    L = X[:, start:-1]
    rr = r[:, start + 1:]
    lev = L * rr - (L - 1) * qd - np.maximum(2 - L, 0) * qq
    part = L * rr + (1 - L) * sg - L * qq
    ret = np.where(L <= 0, sg, np.where(L >= 1, lev, part))
    dX = np.abs(np.diff(X[:, start - 1:], axis=1))[:, :-1]
    g[:, start + 1:] = (1 - tc * dX) * (1 + ret)
    return g


def run(F, reg, Ltgt, start, band=0.0, **kw):
    X = execute(Ltgt, reg, start, band)
    g = daily_factor(X, F["r"], start, **kw)
    E = np.cumprod(g[:, start:], axis=1)
    return E, X


# ------------------------------------------------------------------ plan (withdrawals)
def plan(g, start, balance=760_000, floor=6500, pct=0.007, goal=1_500_000, every=21):
    """Monthly withdrawals of max(floor, pct*equity), sold from the portfolio."""
    n, N = g.shape
    E = np.full(n, float(balance))
    T = N - start - 1
    yearly = []
    hit = np.full(n, np.nan)
    minE = E.copy()
    for k in range(T):
        E = E * g[:, start + 1 + k]
        if (k + 1) % every == 0:
            E = E - np.maximum(floor, pct * E)
        E = np.maximum(E, 0)
        minE = np.minimum(minE, E)
        if (k + 1) % 252 == 0:
            yearly.append(E.copy())
        newly = np.isnan(hit) & (E >= goal)
        hit[newly] = (k + 1) / 252
    return np.array(yearly).T, hit, minE


# ------------------------------------------------------------------ metrics
def metrics(E, rf=SGOV_A):
    E = np.asarray(E, float).ravel()
    ret = E[1:] / E[:-1] - 1
    yrs = len(ret) / 252
    cagr = E[-1] ** (1 / yrs) - 1
    vol = ret.std() * np.sqrt(252)
    peak = np.maximum.accumulate(E)
    dd = E / peak - 1
    return dict(CAGR=cagr, Vol=vol, Sharpe=(ret.mean() * 252 - rf) / vol,
                MaxDD=dd.min(), Ulcer=np.sqrt(np.mean(dd ** 2)),
                Calmar=cagr / -dd.min())
