"""Monte Carlo test of leverage policies, using the same generator as the v5 dashboard
(post-2005 daily returns, 21-day blocks, 25% of paths start with a 2022-style bear),
plus variants with longer blocks and the full 1999+ return pool.

All policies see the SAME random paths (common random numbers), so differences are
due to the rules, not luck.  Each dynamic rule is compared with the static-leverage
frontier at matched risk."""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lev_engine import load_close, features, regime, execute, daily_factor, plan
import policies as pol

XLSX = os.environ.get("QQQ_XLSX", "QQQ-history.xlsx")
OUT = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
N_PATHS = int(os.environ.get("N_PATHS", 2000))
YEARS = 10
SEED_LEN = 300

s = load_close(XLSX)
c, dates = s.values, s.index
assert c[-SEED_LEN:].max() == c.max(), "seed window must contain the all-time high"
rets = np.r_[np.nan, c[1:] / c[:-1] - 1]

# regime state as of the day before the last close (the last close is replayed in-sim)
Fh = features(c[None, :-1])
_, init = regime(Fh, 199)


def make_paths(pool_from="2005-01-01", block=21, bear_prob=0.25, seed=42):
    rng = np.random.default_rng(seed)
    pool = rets[(dates >= pool_from)]
    pool = pool[~np.isnan(pool)]
    bear = rets[(dates >= "2021-11-01") & (dates <= "2022-12-31")]
    nd = 252 * YEARS
    nb = nd // block + 2
    R = np.empty((N_PATHS, nd))
    for p in range(N_PATHS):
        st = rng.integers(0, len(pool) - block, nb)
        r = np.concatenate([pool[k:k + block] for k in st])[:nd]
        if rng.random() < bear_prob:
            bs = rng.integers(0, len(bear) - 21, 7)
            r[:126] = np.concatenate([bear[k:k + 21] for k in bs])[:126]
        R[p] = r
    path = c[-1] * np.cumprod(1 + R, axis=1)
    return np.concatenate([np.tile(c[-SEED_LEN:], (N_PATHS, 1)), path], axis=1)


def candidates(F, reg):
    C = {}
    for L in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
        C[f"static {L:.1f}"] = (pol.static(F, L), 0.0)
    C["ladder 1.3→1.5 @-10% (ATH, reset at high)"] = (pol.ladder(F, 1.3, [(0.10, 1.5)], "ath", True, 0.0), 0.1)
    C["ladder 1.3→1.4 @-5% →1.5 @-10%"] = (pol.ladder(F, 1.3, [(0.05, 1.4), (0.10, 1.5)], "ath", True, 0.0), 0.1)
    C["ladder 1.3→1.6 @-10% →1.9 @-20% (52w)"] = (pol.ladder(F, 1.3, [(0.10, 1.6), (0.20, 1.9)], "52w", True, 0.0), 0.1)
    C["delever highs 1.0 / 1.4@-5% / 1.8@-10% (52w)"] = (pol.ladder(F, 1.0, [(0.05, 1.4), (0.10, 1.8)], "52w", True, 0.0), 0.1)
    C["SMA steps A 2.0/1.7/1.4/1.2/1.0"] = (pol.sma_steps(F, [2.0, 1.7, 1.4, 1.2, 1.0]), 0.1)
    C["SMA steps B 1.8/1.6/1.4/1.2/1.0"] = (pol.sma_steps(F, [1.8, 1.6, 1.4, 1.2, 1.0]), 0.1)
    C["SMA steps C 1.7/1.5/1.3/1.1/1.0"] = (pol.sma_steps(F, [1.7, 1.5, 1.3, 1.1, 1.0]), 0.1)
    C["SMA steps A, 1.3 below SMA"] = (pol.sma_steps(F, [2.0, 1.7, 1.4, 1.2, 1.0], below=1.3), 0.1)
    C["SMA linear 2.0→1.0 over 0-20%"] = (pol.sma_dist(F, 2.0, 1.0, 0.20), 0.1)
    C["SMA linear 1.6→1.0 + RSI2 boost"] = (pol.sma_dist_os(F, 1.6, 1.0, 0.20, 0.5, 5, 5, 2.0), 0.1)
    C["oversold 1.3 +0.6 (RSI2<5, 5d)"] = (pol.oversold(F, 1.3, 0.6, 5, 5), 0.1)
    C["vol target 32% [1.0,1.6]"] = (pol.voltarget(F, 0.32, 10, 1.0, 1.6), 0.1)
    C["vol target 36% [1.0,2.0]"] = (pol.voltarget(F, 0.36, 20, 1.0, 2.0), 0.1)
    return C


def evaluate(Pm, label):
    t0 = time.time()
    F = features(Pm)
    start = SEED_LEN - 1
    reg, _ = regime(F, start, init=init)
    rows = []
    for name, (Lt, band) in candidates(F, reg).items():
        X = execute(Lt, reg, start, band)
        g = daily_factor(X, F["r"], start)
        E = np.cumprod(g[:, start:], axis=1)
        yrs = (E.shape[1] - 1) / 252
        cagr = E[:, -1] ** (1 / yrs) - 1
        peak = np.maximum.accumulate(E, axis=1)
        dd = E / peak - 1
        mdd = dd.min(axis=1)
        ulc = np.sqrt((dd ** 2).mean(axis=1))
        yearly, hit, minE = plan(g, start)
        fin = yearly[:, -1]
        inm = X[:, start:] > 0
        rows.append(dict(
            generator=label, name=name,
            med_CAGR=np.median(cagr), p10_CAGR=np.percentile(cagr, 10), p90_CAGR=np.percentile(cagr, 90),
            med_MaxDD=np.median(mdd), p90_MaxDD=np.percentile(mdd, 10), mean_Ulcer=ulc.mean(),
            AvgL=(X[:, start:] * inm).sum() / inm.sum(),
            plan_median=np.median(fin), plan_p10=np.percentile(fin, 10), plan_p25=np.percentile(fin, 25),
            plan_p90=np.percentile(fin, 90), ruin=np.mean(fin <= 0), below_start=np.mean(fin < 760_000),
            goal_10y=np.mean(~np.isnan(hit)), dip_below_400k=np.mean(minE < 400_000)))
    print(f"  {label}: {time.time() - t0:.0f}s")
    return pd.DataFrame(rows)


def add_matched(df):
    out = []
    for gname, d in df.groupby("generator", sort=False):
        d = d.copy()
        st = d[d.name.str.startswith("static")]

        def interp(xcol, ycol, sign=1):
            x = sign * st[xcol].values
            o = np.argsort(x)
            return lambda v: np.interp(sign * v, x[o], st[ycol].values[o])

        d["xs_CAGR@Ulcer"] = d.med_CAGR - interp("mean_Ulcer", "med_CAGR")(d.mean_Ulcer)
        d["xs_CAGR@MaxDD"] = d.med_CAGR - interp("med_MaxDD", "med_CAGR", -1)(d.med_MaxDD)
        d["xs_p10CAGR@Ulcer"] = d.p10_CAGR - interp("mean_Ulcer", "p10_CAGR")(d.mean_Ulcer)
        d["plan_xs_median@p10"] = d.plan_median - interp("plan_p10", "plan_median")(d.plan_p10)
        d["static_L_same_Ulcer"] = np.interp(d.mean_Ulcer, st.mean_Ulcer.values,
                                             [float(n.split()[1]) for n in st.name])
        out.append(d)
    return pd.concat(out)


if __name__ == "__main__":
    gens = [
        ("v5 generator (post-2005, 21d blocks)", dict(pool_from="2005-01-01", block=21)),
        ("post-2005, 126d blocks", dict(pool_from="2005-01-01", block=126)),
        ("full 1999+, 21d blocks", dict(pool_from="1999-03-11", block=21)),
        ("full 1999+, 126d blocks", dict(pool_from="1999-03-11", block=126)),
    ]
    res = []
    for label, kw in gens:
        res.append(evaluate(make_paths(**kw), label))
    df = add_matched(pd.concat(res))
    df.to_csv(os.path.join(OUT, "mc_results.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    cols = ["name", "med_CAGR", "p10_CAGR", "med_MaxDD", "mean_Ulcer", "AvgL", "static_L_same_Ulcer",
            "xs_CAGR@Ulcer", "xs_CAGR@MaxDD", "xs_p10CAGR@Ulcer", "plan_median", "plan_p10", "ruin",
            "goal_10y", "plan_xs_median@p10"]
    for gname, d in df.groupby("generator", sort=False):
        print(f"\n=== {gname} ===")
        print(d[cols].round(3).to_string(index=False))
