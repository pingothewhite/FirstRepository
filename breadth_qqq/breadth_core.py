"""Breadth indicators and 1-100 score, copied from the Colab "Market Breadth Monitor"
so the backtest uses exactly the same definitions as the live dashboard."""
import numpy as np
import pandas as pd

MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 2e6
REQUIRE_VOLUME_CONFIRM = True

WEIGHTS = {
    "5-day ratio":        0.10,
    "10-day ratio":       0.10,
    "Quarter (±25%)":     0.15,
    "Month (±25%)":       0.10,
    "34-day (±13%)":      0.10,
    "% above 50dma":      0.15,
    "New highs vs lows":  0.10,
    "Advance/decline":    0.10,
    "50dma trend (5d)":   0.05,
    "S&P trend":          0.05,
}


def compute_breadth(px):
    C, H, L, V = px["Close"], px["High"], px["Low"], px["Volume"]
    H, L, V = H.reindex_like(C), L.reindex_like(C), V.reindex_like(C)
    prev = C.shift(1)
    pct = C / prev - 1
    dv20 = (C * V).rolling(20, min_periods=15).mean()
    elig = C.notna() & prev.notna() & (C >= MIN_PRICE) & (dv20 >= MIN_DOLLAR_VOLUME)
    cnt = lambda m: (m & elig).sum(axis=1)

    volok = (V > V.shift(1)) & (V >= 100_000) if REQUIRE_VOLUME_CONFIRM else pd.DataFrame(True, C.index, C.columns)
    chg = lambda n: C / C.shift(n) - 1
    q, m, d34 = chg(65), chg(20), chg(34)

    tr = np.maximum(H - L, np.maximum((H - prev).abs(), (L - prev).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    sma50 = C.rolling(50, min_periods=50).mean()
    has50 = elig & sma50.notna()

    b = pd.DataFrame(index=C.index)
    b["up4"], b["dn4"] = cnt((pct >= 0.04) & volok), cnt((pct <= -0.04) & volok)
    b["up5"], b["dn5"] = b.up4.rolling(5).sum(), b.dn4.rolling(5).sum()
    b["up10"], b["dn10"] = b.up4.rolling(10).sum(), b.dn4.rolling(10).sum()
    b["r5"], b["r10"] = b.up5 / b.dn5.clip(lower=1), b.up10 / b.dn10.clip(lower=1)
    b["up25q"], b["dn25q"] = cnt(q >= 0.25), cnt(q <= -0.25)
    b["up25m"], b["dn25m"] = cnt(m >= 0.25), cnt(m <= -0.25)
    b["up50m"], b["dn50m"] = cnt(m >= 0.50), cnt(m <= -0.50)
    b["up13"], b["dn13"] = cnt(d34 >= 0.13), cnt(d34 <= -0.13)
    b["ext10"] = cnt((C - sma50) / atr >= 10)
    b["pct50"] = ((C > sma50) & has50).sum(axis=1) / has50.sum(axis=1).clip(lower=1)
    b["adv"], b["dec"] = cnt(pct > 0), cnt(pct < 0)
    b["nh"] = cnt(H >= H.rolling(252, min_periods=252).max())
    b["nl"] = cnt(L <= L.rolling(252, min_periods=252).min())
    b["adv_pct"] = b.adv / (b.adv + b.dec).clip(lower=1)
    b["nh_pct"] = b.nh / (b.nh + b.nl).clip(lower=1)
    b["universe"] = elig.sum(axis=1)
    # 52-week high/low needs a full year of data per stock; track how many stocks qualify
    b["hl_universe"] = (elig & H.rolling(252, min_periods=252).max().notna()).sum(axis=1)

    spx = px["SPX"].reindex(C.index).ffill()
    b["spx"] = spx
    b["spx_sma50"], b["spx_sma200"] = spx.rolling(50).mean(), spx.rolling(200).mean()
    b["spx_hi252"] = spx.rolling(252, min_periods=1).max()

    b = b[b.universe >= 0.5 * b.universe.rolling(60, min_periods=1).median()]
    return b


def score_components(b):
    sig = lambda x: 1 / (1 + np.exp(-x))
    bal = lambda a, c: (a + 1) / (a + c + 2)
    comp = pd.DataFrame(index=b.index)
    comp["5-day ratio"] = sig(2.5 * np.log((b.up5 + 1) / (b.dn5 + 1)))
    comp["10-day ratio"] = sig(2.5 * np.log((b.up10 + 1) / (b.dn10 + 1)))
    comp["Quarter (±25%)"] = bal(b.up25q, b.dn25q)
    comp["Month (±25%)"] = bal(b.up25m, b.dn25m)
    comp["34-day (±13%)"] = bal(b.up13, b.dn13)
    comp["% above 50dma"] = ((b.pct50 - 0.20) / 0.60).clip(0, 1)
    comp["New highs vs lows"] = bal(b.nh.rolling(5).sum(), b.nl.rolling(5).sum())
    adv_share = (b.adv / (b.adv + b.dec).clip(lower=1)).rolling(5).mean()
    comp["Advance/decline"] = ((adv_share - 0.35) / 0.30).clip(0, 1)
    comp["50dma trend (5d)"] = sig(20 * b.pct50.diff(5))
    comp["S&P trend"] = 0.5 * (b.spx > b.spx_sma50) + 0.5 * (b.spx > b.spx_sma200)
    return comp


def add_score(b):
    comp = score_components(b)
    w = pd.Series(WEIGHTS, dtype=float)
    w = w / w.sum()
    raw = (comp[w.index] * w).sum(axis=1, min_count=len(w))
    b = b.copy()
    b["score"] = (1 + 99 * raw).round()
    return b, comp
