"""Robustness scorecard: every candidate rule vs the static-leverage frontier at
matched risk (Ulcer index), across
  * history: full 1999-2026, three eras, and with execution one day late
  * Monte Carlo: 4 generators (v5 post-2005 21d blocks; 126d blocks; full-pool 21d; full-pool 126d)
A rule only "robustly beats static" if its excess return is positive in ALL tests."""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lev_engine import load_close, features, regime, execute, daily_factor, metrics, plan
import policies as pol
import mc_study as mc

OUT = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
STATIC_GRID = [round(x, 2) for x in np.arange(0.8, 2.61, 0.1)]


def build(F, reg, dates=None):
    """name -> (target leverage array, no-trade band)."""
    C = {f"static {L:.1f}": (pol.static(F, L), 0.0) for L in STATIC_GRID}
    C.update({
        # --- your correction ladders
        "U1 ladder 1.3 → 1.5 at -10% (reset at ATH)": (pol.ladder(F, 1.3, [(0.10, 1.5)], "ath", True, 0.0), 0.1),
        "U2 ladder 1.3 → 1.4 at -5% → 1.5 at -10%": (pol.ladder(F, 1.3, [(0.05, 1.4), (0.10, 1.5)], "ath", True, 0.0), 0.1),
        "U3 ladder 1.4 → 1.6 at -10%": (pol.ladder(F, 1.4, [(0.10, 1.6)], "ath", True, 0.0), 0.1),
        "U4 delever highs 1.2 / 1.4 at -5% / 1.6 at -10%": (pol.ladder(F, 1.2, [(0.05, 1.4), (0.10, 1.6)], "ath", True, 0.0), 0.1),
        "U5 ladder 1.3 → 1.6 at -10% → 1.9 at -20% (52w)": (pol.ladder(F, 1.3, [(0.10, 1.6), (0.20, 1.9)], "52w", True, 0.0), 0.1),
        # --- distance from the 200-day SMA
        "S1 SMA steps 2.0/1.7/1.4/1.2/1.0": (pol.sma_steps(F, [2.0, 1.7, 1.4, 1.2, 1.0]), 0.1),
        "S2 SMA steps 1.8/1.6/1.4/1.2/1.0": (pol.sma_steps(F, [1.8, 1.6, 1.4, 1.2, 1.0]), 0.1),
        "S3 SMA steps 1.7/1.5/1.3/1.1/1.0": (pol.sma_steps(F, [1.7, 1.5, 1.3, 1.1, 1.0]), 0.1),
        # --- volatility targeting
        "V1 vol target 32% (1.0–1.6x)": (pol.voltarget(F, 0.32, 10, 1.0, 1.6), 0.1),
        "V2 vol target 36% (1.0–2.0x)": (pol.voltarget(F, 0.36, 20, 1.0, 2.0), 0.1),
        "V3 vol target 28% (1.0–1.6x)": (pol.voltarget(F, 0.28, 20, 1.0, 1.6), 0.1),
        "V4 vol target 30% (1.0–1.8x)": (pol.voltarget(F, 0.30, 20, 1.0, 1.8), 0.1),
        # --- short-term oversold
        "O1 RSI2<5 boost: 1.3 +0.6 for 5d": (pol.oversold(F, 1.3, 0.6, 5, 5), 0.1),
        "O2 RSI2<10 boost: 1.3 +0.5 for 5d": (pol.oversold(F, 1.3, 0.5, 10, 5), 0.1),
        "O3 RSI2<5 boost: 1.4 +0.4 for 5d": (pol.oversold(F, 1.4, 0.4, 5, 5), 0.1),
        # --- combinations
        "C1 SMA linear 1.6→1.0 + RSI2 boost": (pol.sma_dist_os(F, 1.6, 1.0, 0.20, 0.5, 5, 5, 2.0), 0.1),
        "C2 vol target 32% + RSI2 boost": (pol.vt_os(F, 0.32, 10, 1.0, 1.6, 0.4), 0.1),
        "C3 SMA steps, capped by 36% vol target": (pol.sma_volcap(F, [2.0, 1.7, 1.4, 1.2, 1.0], 0.36), 0.1),
        "C4 blend SMA steps + vol target 32%": (pol.blend(pol.sma_steps(F, [1.8, 1.6, 1.4, 1.2, 1.0]),
                                                           pol.voltarget(F, 0.32, 10, 1.0, 1.6)), 0.1),
        "C5 vol target 36% + RSI2 boost": (pol.vt_os(F, 0.36, 20, 1.0, 2.0, 0.4), 0.1),
    })
    if dates is not None:
        C["T1 turn-of-month 1.3 +0.5 (history only)"] = (pol.turn_of_month(dates, F["P"].shape, 1.3, 0.5), 0.1)
    return C


def frontier_excess(d, risk="Ulcer", ret="CAGR"):
    st = d[d.index.str.startswith("static")]
    o = np.argsort(st[risk].values)
    return d[ret] - np.interp(d[risk], st[risk].values[o], st[ret].values[o])


def history_tests(s):
    P = s.values[None, :]
    F = features(P)
    start = 199
    reg, _ = regime(F, start)
    D = s.index[start:]
    eras = {"hist 2000-08": ("1999-12-01", "2008-12-31"), "hist 2009-16": ("2009-01-01", "2016-12-31"),
            "hist 2017-26": ("2017-01-01", "2026-12-31")}
    res = {}
    for name, (Lt, band) in build(F, reg, s.index).items():
        X = execute(Lt, reg, start, band)
        for lag in (0, 1):
            Xl = X if lag == 0 else np.concatenate([np.zeros((1, 1)), X[:, :-1]], axis=1)
            E = np.cumprod(daily_factor(Xl, F["r"], start)[:, start:], axis=1).ravel()
            key = "hist full" if lag == 0 else "hist full, 1-day late"
            m = metrics(E)
            res.setdefault(key, {})[name] = dict(CAGR=m["CAGR"], Ulcer=m["Ulcer"], MaxDD=m["MaxDD"],
                                                 AvgL=X[0, start:][X[0, start:] > 0].mean())
            if lag == 0:
                for era, (a, b) in eras.items():
                    i0, i1 = np.searchsorted(D, pd.Timestamp(a)), np.searchsorted(D, pd.Timestamp(b), "right")
                    e = E[i0:i1] / E[i0]
                    mm = metrics(e)
                    res.setdefault(era, {})[name] = dict(CAGR=mm["CAGR"], Ulcer=mm["Ulcer"], MaxDD=mm["MaxDD"])
    return {k: pd.DataFrame(v).T for k, v in res.items()}


def mc_tests():
    out, plans = {}, {}
    gens = [("MC v5 (post-2005, 21d)", dict(pool_from="2005-01-01", block=21)),
            ("MC post-2005, 126d", dict(pool_from="2005-01-01", block=126)),
            ("MC full pool, 21d", dict(pool_from="1999-03-11", block=21)),
            ("MC full pool, 126d", dict(pool_from="1999-03-11", block=126))]
    for label, kw in gens:
        Pm = mc.make_paths(**kw)
        F = features(Pm)
        start = mc.SEED_LEN - 1
        reg, _ = regime(F, start, init=mc.init)
        rows, prow = {}, {}
        for name, (Lt, band) in build(F, reg).items():
            X = execute(Lt, reg, start, band)
            g = daily_factor(X, F["r"], start)
            E = np.cumprod(g[:, start:], axis=1)
            yrs = (E.shape[1] - 1) / 252
            cagr = E[:, -1] ** (1 / yrs) - 1
            dd = E / np.maximum.accumulate(E, axis=1) - 1
            rows[name] = dict(CAGR=np.median(cagr), Ulcer=np.sqrt((dd ** 2).mean(axis=1)).mean(),
                              MaxDD=np.median(dd.min(axis=1)), p10CAGR=np.percentile(cagr, 10))
            yearly, hit, _ = plan(g, start)
            fin = yearly[:, -1]
            prow[name] = dict(median_end=np.median(fin), p25_end=np.percentile(fin, 25),
                              ruin=np.mean(fin <= 0), goal=np.mean(~np.isnan(hit)))
        out[label] = pd.DataFrame(rows).T
        plans[label] = pd.DataFrame(prow).T
        print("  done", label)
    return out, plans


if __name__ == "__main__":
    s = load_close(os.environ.get("QQQ_XLSX", "QQQ-history.xlsx"))
    tests = history_tests(s)
    mct, plans = mc_tests()
    tests.update(mct)
    xs = pd.DataFrame({k: frontier_excess(v) for k, v in tests.items()})
    dyn = xs[~xs.index.str.startswith("static")].copy()
    mc_cols = [c for c in dyn.columns if c.startswith("MC")]
    hist_cols = [c for c in dyn.columns if c.startswith("hist")]
    dyn["worst (all tests)"] = dyn[hist_cols + mc_cols].min(axis=1)
    dyn["tests won"] = (dyn[hist_cols + mc_cols] > 0).sum(axis=1).astype(str) + f"/{len(hist_cols + mc_cols)}"
    dyn = dyn.sort_values("worst (all tests)", ascending=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    print("\nExcess CAGR vs static leverage at the SAME risk (Ulcer), percentage points per year:")
    print((dyn.drop(columns="tests won") * 100).round(2).assign(won=dyn["tests won"]).to_string())
    h = tests["hist full"]
    print("\nFull-history detail:")
    print(h.loc[[i for i in h.index if not i.startswith("static") or i in
                 ("static 1.0", "static 1.3", "static 1.4", "static 1.5", "static 1.6")]].round(3).to_string())
    print("\nYour plan ($760k, $6.5k/mo floor or 0.7%) in the v5 generator, 10 years:")
    pv = plans["MC v5 (post-2005, 21d)"]
    print(pv.round(3).to_string())
    print("\nSame plan, full-pool generator (includes 2000-02):")
    print(plans["MC full pool, 21d"].round(3).to_string())
    dyn.to_csv(os.path.join(OUT, "scorecard.csv"))
    pd.concat(tests, names=["test"]).to_csv(os.path.join(OUT, "scorecard_tests.csv"))
    pd.concat(plans, names=["generator"]).to_csv(os.path.join(OUT, "scorecard_plans.csv"))
