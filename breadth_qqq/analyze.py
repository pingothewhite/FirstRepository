"""Does the breadth dashboard predict QQQ?  Reads data/breadth_history.csv + QQQ-history.xlsx,
writes data/results.json (all numbers used in the report).

Tests, in order of how much they can be trusted:
  1. Information coefficients: rank correlation of each indicator with QQQ's forward return,
     Newey-West t-stats (overlapping returns), with and without controlling for QQQ's own past.
  2. Buckets: forward returns by quintile of each indicator.
  3. Event studies for the dashboard's flags, vs. random dates.
  4. Granger causality, both directions, on weekly data.
  5. Walk-forward (out-of-sample) regressions and simple timing rules traded with a 1-day lag.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
HORIZONS = [5, 10, 21, 63]
START = "2016-09-27"
RNG = np.random.default_rng(7)


# ---------------------------------------------------------------- data
def load_qqq():
    q = pd.read_excel(os.path.join(HERE, "QQQ-history.xlsx"))
    q["Date"] = pd.to_datetime(q["Date"]).dt.normalize()
    q = q.set_index("Date").sort_index()
    return q.rename(columns={"Adj. Close": "adj"})


def load_breadth():
    b = pd.read_csv(os.path.join(DATA, "breadth_history.csv"), parse_dates=["Date"], index_col="Date")
    return b


FEATURES = {
    # name: (label, sign that the dashboard treats as bullish)
    "score":      ("Breadth score (1-100)", +1),
    "r5":         ("5-day 4% up/down ratio", +1),
    "r10":        ("10-day 4% up/down ratio", +1),
    "up4_f":      ("% of stocks up 4%+ today", +1),
    "dn4_f":      ("% of stocks down 4%+ today", -1),
    "net25q":     ("Net up-25% on the quarter", +1),
    "net25m":     ("Net up-25% on the month", +1),
    "up50m_f":    ("% up 50%+ on the month", +1),
    "dn50m_f":    ("% down 50%+ on the month", -1),
    "net13":      ("Net up-13% over 34 days", +1),
    "ext10_f":    ("% 10x ATR extended", -1),
    "pct50":      ("% above 50dma", +1),
    "pct50_d5":   ("5-day change in % above 50dma", +1),
    "adv5":       ("Adv % (5-day avg)", +1),
    "adv_pct":    ("Adv % (1 day)", +1),
    "nh_pct5":    ("New-high share (5-day)", +1),
    "net_hl":     ("Net new highs % of universe", +1),
}


def build_features(b):
    f = pd.DataFrame(index=b.index)
    u = b.universe.clip(lower=1)
    hu = b.hl_universe.clip(lower=1)
    f["score"] = b.score
    f["r5"], f["r10"] = np.log(b.r5.clip(lower=0.05)), np.log(b.r10.clip(lower=0.05))
    f["up4_f"], f["dn4_f"] = b.up4 / u, b.dn4 / u
    f["net25q"] = (b.up25q - b.dn25q) / u
    f["net25m"] = (b.up25m - b.dn25m) / u
    f["up50m_f"], f["dn50m_f"] = b.up50m / u, b.dn50m / u
    f["net13"] = (b.up13 - b.dn13) / u
    f["ext10_f"] = b.ext10 / u
    f["pct50"] = b.pct50
    f["pct50_d5"] = b.pct50.diff(5)
    f["adv5"] = b.adv_pct.rolling(5).mean()
    f["adv_pct"] = b.adv_pct
    nh5, nl5 = b.nh.rolling(5).sum(), b.nl.rolling(5).sum()
    f["nh_pct5"] = (nh5 + 1) / (nh5 + nl5 + 2)
    f["net_hl"] = (b.nh - b.nl) / hu
    for c in [c for c in b.columns if c.startswith("c_")]:
        f[c] = b[c]
    return f


def forward(logp, h):
    return logp.shift(-h) - logp


# ---------------------------------------------------------------- stats helpers
def nw_slope(y, X, lags):
    X = sm.add_constant(X)
    m = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m


def spearman(a, b):
    ok = a.notna() & b.notna()
    return a[ok].rank().corr(b[ok].rank())


def zscore(s):
    return (s - s.mean()) / s.std()


def perf(ret):
    """ret: daily simple returns of a strategy."""
    ret = ret.dropna()
    eq = (1 + ret).cumprod()
    yrs = len(ret) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = ret.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "maxdd": dd, "final": eq.iloc[-1]}


def decluster(mask, gap):
    """First day of each run of signals, ignoring repeats within `gap` trading days."""
    idx, last = [], -10**9
    for i, v in enumerate(mask.values):
        if v and i - last > gap:
            idx.append(i)
            last = i
        elif v:
            pass
    return mask.index[idx]


# ---------------------------------------------------------------- analysis
def main():
    q = load_qqq()
    b = load_breadth()
    feats = build_features(b)
    df = feats.join(q[["adj", "Close"]], how="inner")
    df = df[df.index >= START]
    lp = np.log(df.adj)
    for h in HORIZONS:
        df[f"f{h}"] = forward(lp, h)
    df["p21"] = lp - lp.shift(21)          # QQQ's own past, as a control
    df["p63"] = lp - lp.shift(63)
    df["p252"] = lp - lp.shift(252)
    df["vol21"] = lp.diff().rolling(21).std() * np.sqrt(252)
    dret = df.adj.pct_change()
    R = {"meta": {}, "current": {}, "ic": [], "ic_ctrl": [], "ic_sub": [], "quint": {}, "events": [],
         "states": [], "granger": [], "oos": [], "strategies": [], "series": {}}

    n_days = int(df.adj.notna().sum())
    R["meta"] = {
        "start": f"{df.index[0]:%Y-%m-%d}", "end": f"{df.index[-1]:%Y-%m-%d}", "days": n_days,
        "universe_first": int(b.universe.loc[df.index[0]]), "universe_last": int(b.universe.loc[df.index[-1]]),
        "universe_median": float(b.universe.loc[df.index].median()),
        "qqq_first": float(df.Close.iloc[0]), "qqq_last": float(df.Close.iloc[-1]),
        "qqq_cagr": float((df.adj.iloc[-1] / df.adj.iloc[0]) ** (252 / (len(df) - 1)) - 1),
        "breadth_end": f"{b.index[-1]:%Y-%m-%d}",
    }
    base = {h: {"mean": float(df[f"f{h}"].mean()), "pos": float((df[f"f{h}"] > 0).mean()),
                "median": float(df[f"f{h}"].median())} for h in HORIZONS}
    R["baseline"] = base

    # contemporaneous vs predictive: same-day correlation of breadth with QQQ's daily move
    R["meta"]["same_day_corr_adv"] = float(df.adv_pct.corr(lp.diff()))
    R["meta"]["same_day_corr_dn4"] = float(df.dn4_f.corr(lp.diff()))
    R["meta"]["next_day_corr_adv"] = float(df.adv_pct.corr(lp.diff().shift(-1)))

    # 1. information coefficients --------------------------------------------------------
    mid = df.index[len(df) // 2]
    R["meta"]["split"] = f"{mid:%Y-%m-%d}"
    names = list(FEATURES) + [c for c in df.columns if c.startswith("c_")]
    for name in names:
        x = df[name]
        for h in HORIZONS:
            y = df[f"f{h}"]
            ok = x.notna() & y.notna()
            m = nw_slope(y[ok], zscore(x[ok]), h)
            mc = nw_slope(y[ok & df.p63.notna()],
                          pd.concat([zscore(x), zscore(df.p21), zscore(df.p63), zscore(df.vol21)], axis=1)
                          [ok & df.p63.notna()].set_axis(["x", "p21", "p63", "vol"], axis=1), h)
            R["ic"].append({"feature": name, "h": h, "ic": spearman(x, y), "t": float(m.tvalues.iloc[1]),
                            "beta": float(m.params.iloc[1]), "r2": float(m.rsquared), "n": int(ok.sum())})
            R["ic_ctrl"].append({"feature": name, "h": h, "t": float(mc.tvalues["x"]),
                                 "beta": float(mc.params["x"])})
            R["ic_sub"].append({"feature": name, "h": h,
                                "ic1": spearman(x[:mid], y[:mid]), "ic2": spearman(x[mid:], y[mid:])})

    # 2. quintile buckets (full-sample cut points; see walk-forward section for no-look-ahead)
    for name in list(FEATURES):
        x = df[name]
        ok = x.notna()
        try:
            qb = pd.qcut(x[ok].rank(method="first"), 5, labels=False)
        except ValueError:
            continue
        rows = []
        for k in range(5):
            sel = qb[qb == k].index
            lo, hi = x.loc[sel].min(), x.loc[sel].max()
            row = {"q": k + 1, "lo": float(lo), "hi": float(hi), "n": int(len(sel))}
            for h in HORIZONS:
                y = df.loc[sel, f"f{h}"].dropna()
                row[f"m{h}"] = float(y.mean())
                row[f"p{h}"] = float((y > 0).mean())
            rows.append(row)
        R["quint"][name] = rows

    # 3. event studies -------------------------------------------------------------------
    u = b.universe.reindex(df.index)
    ema_adv = b.adv_pct.ewm(span=10, adjust=False).mean().reindex(df.index)
    zweig = (ema_adv > 0.615) & (ema_adv.rolling(10).min() < 0.40)
    events = {
        "Overbought: 10-day ratio >= 2": b.r10.reindex(df.index) >= 2,
        "Overbought: 5-day ratio >= 2": b.r5.reindex(df.index) >= 2,
        "Oversold: 5-day ratio <= 0.5": b.r5.reindex(df.index) <= 0.5,
        "Oversold: 10-day ratio <= 0.5": b.r10.reindex(df.index) <= 0.5,
        "Heavy selling: 10%+ of stocks down 4%": b.dn4.reindex(df.index) >= 0.10 * u,
        "Breadth thrust: 10%+ of stocks up 4%": b.up4.reindex(df.index) >= 0.10 * u,
        "Washed out: <= 20% above 50dma": df.pct50 <= 0.20,
        "Stretched: >= 80% above 50dma": df.pct50 >= 0.80,
        "Froth: many stocks 10x ATR extended": b.ext10.reindex(df.index) >= np.maximum(10, 0.005 * u),
        "Divergence: S&P near high, < 40% above 50dma":
            (b.spx >= 0.97 * b.spx_hi252).reindex(df.index) & (df.pct50 < 0.40),
        "Zweig breadth thrust": zweig,
        "Score <= 20 (strongly bearish)": df.score <= 20,
        "Score >= 80 (strongly bullish)": df.score >= 80,
        "Score crosses up through 50": (df.score >= 50) & (df.score.shift(1) < 50),
        "Score crosses down through 50": (df.score < 50) & (df.score.shift(1) >= 50),
    }
    valid = df.index[df.f63.notna()]
    for label, mask in events.items():
        mask = mask.fillna(False)
        dates = decluster(mask, 20)
        row = {"event": label, "days_on": int(mask.sum()), "n": int(len(dates)),
               "dates": [f"{d:%Y-%m-%d}" for d in dates],
               "on_now": bool(mask.iloc[-1]) if len(mask) else False,
               "last": f"{mask[mask].index[-1]:%Y-%m-%d}" if mask.any() else None}
        for h in HORIZONS:
            y = df.loc[dates, f"f{h}"].dropna()
            row[f"m{h}"] = float(y.mean()) if len(y) else None
            row[f"med{h}"] = float(y.median()) if len(y) else None
            row[f"p{h}"] = float((y > 0).mean()) if len(y) else None
            row[f"n{h}"] = int(len(y))
            if len(y) >= 3:
                pool = df[f"f{h}"].dropna().values
                sims = RNG.choice(pool, size=(20000, len(y))).mean(axis=1)
                # two-sided: how often random dates give an average this far from the norm
                dev = abs(y.mean() - pool.mean())
                row[f"pv{h}"] = float((abs(sims - pool.mean()) >= dev).mean())
            else:
                row[f"pv{h}"] = None
        # all signal days (not declustered), 21-day, with NW t vs. other days
        y = df.f21
        ok = y.notna()
        m = nw_slope(y[ok], mask[ok].astype(float), 21)
        row["all_days_diff21"] = float(m.params.iloc[1])
        row["all_days_t21"] = float(m.tvalues.iloc[1])
        R["events"].append(row)

    # regime states: forward returns while a condition holds vs. not
    states = {
        "Primary trend weak (more stocks down 25% on qtr than up)": b.dn25q > b.up25q,
        "Intermediate trend weak (down-13% > 2x up-13%)": b.dn13 > 2 * b.up13,
        "Score >= 60": b.score >= 60,
        "Score < 40": b.score < 40,
        "% above 50dma < 40%": b.pct50 < 0.40,
        "% above 50dma rising over 5 days": b.pct50.diff(5) > 0,
        "S&P above 200dma": b.spx > b.spx_sma200,
    }
    for label, st in states.items():
        st = st.reindex(df.index).fillna(False)
        row = {"state": label, "share": float(st.mean()), "on_now": bool(st.iloc[-1])}
        for h in [5, 21, 63]:
            y = df[f"f{h}"]
            ok = y.notna()
            m = nw_slope(y[ok], st[ok].astype(float), h)
            row[f"on{h}"] = float(y[ok & st].mean())
            row[f"off{h}"] = float(y[ok & ~st].mean())
            row[f"pon{h}"] = float((y[ok & st] > 0).mean())
            row[f"poff{h}"] = float((y[ok & ~st] > 0).mean())
            row[f"t{h}"] = float(m.tvalues.iloc[1])
        # daily vol of QQQ next 21 days when on vs off (risk, not direction)
        fv = lp.diff().rolling(21).std().shift(-21) * np.sqrt(252)
        row["vol_on"] = float(fv[st].mean())
        row["vol_off"] = float(fv[~st].mean())
        R["states"].append(row)

    # 4. Granger causality on weekly data ------------------------------------------------
    wk = pd.DataFrame({
        "qqq": lp.resample("W-FRI").last().diff(),
        "score": df.score.resample("W-FRI").last().diff(),
        "pct50": df.pct50.resample("W-FRI").last().diff(),
        "net25q": df.net25q.resample("W-FRI").last().diff(),
        "net13": df.net13.resample("W-FRI").last().diff(),
        "adv": df.adv_pct.resample("W-FRI").mean() - 0.5,
        "net_hl": df.net_hl.resample("W-FRI").mean(),
        "r10": df.r10.resample("W-FRI").last(),
    }).dropna()
    R["meta"]["weeks"] = int(len(wk))
    for var in [c for c in wk.columns if c != "qqq"]:
        res = {}
        for direction, cols in [("breadth->qqq", ["qqq", var]), ("qqq->breadth", [var, "qqq"])]:
            g = grangercausalitytests(wk[cols], maxlag=4)
            res[direction] = {lag: float(g[lag][0]["ssr_ftest"][1]) for lag in (1, 2, 4)}
        R["granger"].append({"var": var, **res})
    # daily too (lags 1 and 5), for the headline variables
    dd = pd.DataFrame({"qqq": lp.diff(), "adv": df.adv_pct - 0.5, "score": df.score.diff(),
                       "pct50": df.pct50.diff(), "dn4": df.dn4_f, "up4": df.up4_f}).dropna()
    R["granger_daily"] = []
    for var in ["adv", "score", "pct50", "dn4", "up4"]:
        out = {"var": var}
        for direction, cols in [("breadth->qqq", ["qqq", var]), ("qqq->breadth", [var, "qqq"])]:
            g = grangercausalitytests(dd[cols], maxlag=5)
            out[direction] = {lag: float(g[lag][0]["ssr_ftest"][1]) for lag in (1, 5)}
        R["granger_daily"].append(out)

    # 5. walk-forward -------------------------------------------------------------------
    # Refit every month on data whose 21-day outcome was already known; predict next month.
    h = 21
    y = df[f"f{h}"]
    models = {
        "Breadth score": ["score"],
        "% above 50dma": ["pct50"],
        "Net 25% quarter": ["net25q"],
        "Net 13% / 34 days": ["net13"],
        "New-high share": ["nh_pct5"],
        "QQQ momentum only (63d + 252d)": ["p63", "p252"],
        "Breadth (score, pct50, net25q, net_hl) + momentum": ["score", "pct50", "net25q", "net_hl", "p63", "p252"],
    }
    first_test = df.index[0] + pd.DateOffset(years=3)
    month_starts = df.loc[first_test:].resample("MS").first().index
    month_starts = [df.index[df.index.searchsorted(m)] for m in month_starts if m <= df.index[-1]]
    preds = {k: pd.Series(np.nan, df.index) for k in models}
    preds["Historical mean"] = pd.Series(np.nan, df.index)
    for i, t in enumerate(month_starts):
        nxt = month_starts[i + 1] if i + 1 < len(month_starts) else df.index[-1] + pd.Timedelta(days=1)
        pos = df.index.get_loc(t)
        train = df.iloc[: max(pos - h, 0)]          # outcomes fully observed before t
        test = df.loc[(df.index >= t) & (df.index < nxt)]
        preds["Historical mean"].loc[test.index] = train[f"f{h}"].mean()
        for k, cols in models.items():
            tr = train[cols + [f"f{h}"]].dropna()
            if len(tr) < 250:
                continue
            m = sm.OLS(tr[f"f{h}"], sm.add_constant(tr[cols])).fit()
            preds[k].loc[test.index] = m.predict(sm.add_constant(test[cols], has_constant="add"))
    ok = y.notna() & preds["Historical mean"].notna()
    for k in models:
        ok &= preds[k].notna()
    sse0 = ((y[ok] - preds["Historical mean"][ok]) ** 2).sum()
    for k in models:
        e = y[ok] - preds[k][ok]
        r2 = 1 - (e ** 2).sum() / sse0
        hit = float((np.sign(preds[k][ok] - preds["Historical mean"][ok]) ==
                     np.sign(y[ok] - preds["Historical mean"][ok])).mean())
        corr = float(preds[k][ok].corr(y[ok]))
        R["oos"].append({"model": k, "r2": float(r2), "dir_hit": hit, "corr": corr})
    R["meta"]["oos_start"] = f"{y[ok].index[0]:%Y-%m-%d}"
    R["meta"]["oos_n"] = int(ok.sum())

    # timing rules: signal known at close t, position held from close t+1 (1-day execution lag)
    lag = 2   # position on day t+2 return = close(t+1) -> close(t+2)
    tests = df.index >= first_test
    rules = {
        "Buy & hold QQQ": pd.Series(1.0, df.index),
        "In when score >= 40": (df.score >= 40).astype(float),
        "In when score >= 50": (df.score >= 50).astype(float),
        "Out when score < 20 (only extreme bear)": (df.score >= 20).astype(float),
        "In when % above 50dma >= 40%": (df.pct50 >= 0.40).astype(float),
        "Out when primary trend weak (dn25q > up25q)": (b.dn25q <= b.up25q).reindex(df.index).astype(float),
        "In when S&P > 200dma": (b.spx > b.spx_sma200).reindex(df.index).astype(float),
        "In when QQQ > its 200dma": (df.adj > df.adj.rolling(200).mean()).astype(float),
        "Walk-forward model says above-average": (preds["Breadth (score, pct50, net25q, net_hl) + momentum"] >=
                                                  preds["Historical mean"]).astype(float),
    }
    for k, pos in rules.items():
        held = pos.shift(lag).fillna(0)
        r = (held * dret)[tests]
        p = perf(r)
        p.update({"rule": k, "invested": float(held[tests].mean()),
                  "switches": int((held[tests].diff().abs() > 0).sum())})
        R["strategies"].append(p)
    R["meta"]["strategy_start"] = f"{df.index[tests][0]:%Y-%m-%d}"
    eqs = {}
    for k in ["Buy & hold QQQ", "In when score >= 40", "Out when primary trend weak (dn25q > up25q)",
              "In when QQQ > its 200dma"]:
        held = rules[k].shift(lag).fillna(0)
        eqs[k] = (1 + (held * dret)[tests].fillna(0)).cumprod()

    # current snapshot -------------------------------------------------------------------
    last = df.index[-1]
    cur = {}
    for name in list(FEATURES) + [c for c in df.columns if c.startswith("c_")]:
        v = df[name].iloc[-1]
        cur[name] = {"value": float(v), "pctile": float((df[name].dropna() < v).mean())}
    R["current"] = {"date": f"{last:%Y-%m-%d}", "features": cur,
                    "raw": {k: float(b.loc[last, k]) for k in
                            ["score", "up4", "dn4", "r5", "r10", "up25q", "dn25q", "up25m", "dn25m",
                             "up50m", "dn50m", "up13", "dn13", "ext10", "pct50", "adv", "dec", "adv_pct",
                             "nh", "nl", "nh_pct", "universe", "spx", "spx_hi252"]},
                    "qqq_p63": float(df.p63.iloc[-1]), "qqq_p252": float(df.p252.iloc[-1]),
                    "qqq_vs_200": float(df.adj.iloc[-1] / df.adj.rolling(200).mean().iloc[-1] - 1)}
    # analogs: days with a similar score, % above 50dma and net-25% reading
    z = df[["score", "pct50", "net25q"]].dropna()
    zz = (z - z.mean()) / z.std()
    dist = np.sqrt(((zz - zz.iloc[-1]) ** 2).sum(axis=1))
    dist = dist[dist.index <= last - pd.Timedelta(days=100)]
    near = dist.nsmallest(400)
    picked = decluster(pd.Series(True, near.sort_index().index).reindex(df.index).fillna(False), 20)
    picked = [d for d in picked if d in near.index][:25]
    an = df.loc[picked, ["score", "pct50", "net25q", "f21", "f63"]].copy()
    R["analogs"] = {"n": len(an), "m21": float(an.f21.mean()), "p21": float((an.f21 > 0).mean()),
                    "m63": float(an.f63.dropna().mean()), "p63": float((an.f63.dropna() > 0).mean()),
                    "rows": [{"date": f"{d:%Y-%m-%d}", **{k: (None if pd.isna(v) else float(v)) for k, v in r.items()}}
                             for d, r in an.iterrows()]}

    # series for charts (weekly to keep it light)
    w = df[["Close", "score", "pct50", "net25q"]].resample("W-FRI").last().dropna()
    R["series"]["weekly"] = {"date": [f"{d:%Y-%m-%d}" for d in w.index], "qqq": w.Close.round(2).tolist(),
                             "score": w.score.round(0).tolist(), "pct50": (w.pct50 * 100).round(1).tolist(),
                             "net25q": (w.net25q * 100).round(1).tolist()}
    ew = pd.DataFrame(eqs).resample("W-FRI").last()
    R["series"]["equity"] = {"date": [f"{d:%Y-%m-%d}" for d in ew.index],
                             **{k: ew[k].round(4).tolist() for k in ew.columns}}

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else round(float(o), 6)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    with open(os.path.join(DATA, "results.json"), "w") as f:
        json.dump(clean(R), f)
    print_summary(R)


def print_summary(R):
    m = R["meta"]
    print(f"\n{m['start']} -> {m['end']}  ({m['days']} days, universe {m['universe_first']}->{m['universe_last']})")
    print(f"same-day corr(adv%, QQQ) {m['same_day_corr_adv']:.2f}   next-day {m['next_day_corr_adv']:.3f}")
    ic = pd.DataFrame(R["ic"]).merge(pd.DataFrame(R["ic_ctrl"]), on=["feature", "h"], suffixes=("", "_ctrl"))
    ic = ic.merge(pd.DataFrame(R["ic_sub"]), on=["feature", "h"])
    print("\nInformation coefficients (Spearman) & Newey-West t (raw / controlling for QQQ momentum+vol)")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(ic.pivot(index="feature", columns="h", values="ic").round(3))
        print(ic.pivot(index="feature", columns="h", values="t").round(2))
        print(ic.pivot(index="feature", columns="h", values="t_ctrl").round(2))
        print(ic[ic.h == 21][["feature", "ic1", "ic2"]].round(3).to_string())
        ev = pd.DataFrame(R["events"])
        print(ev[["event", "n", "m5", "m21", "m63", "p21", "p63", "pv21", "pv63", "all_days_t21", "on_now"]]
              .round(3).to_string())
        print("baseline", {h: round(v["mean"], 4) for h, v in R["baseline"].items()})
        print(pd.DataFrame(R["states"]).round(3).to_string())
        print(pd.DataFrame([{"var": g["var"], **{f"b>q {k}": v for k, v in g["breadth->qqq"].items()},
                             **{f"q>b {k}": v for k, v in g["qqq->breadth"].items()}} for g in R["granger"]])
              .round(3).to_string())
        print(pd.DataFrame([{"var": g["var"], **{f"b>q {k}": v for k, v in g["breadth->qqq"].items()},
                             **{f"q>b {k}": v for k, v in g["qqq->breadth"].items()}} for g in R["granger_daily"]])
              .round(4).to_string())
        print(pd.DataFrame(R["oos"]).round(4).to_string())
        print(pd.DataFrame(R["strategies"]).round(3).to_string())
        print("analogs", {k: v for k, v in R["analogs"].items() if k != "rows"})
        print("current", R["current"]["raw"])


if __name__ == "__main__":
    main()
