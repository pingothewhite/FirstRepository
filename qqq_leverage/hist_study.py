"""Historical backtest (Dec-1999 → Sep-2026) of many leverage policies layered on the
v5 regime filter, scored against the static-leverage frontier at matched risk."""
import itertools, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lev_engine import load_close, features, regime, run, metrics
import policies as pol

XLSX = os.environ.get("QQQ_XLSX", "QQQ-history.xlsx")
OUT = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
BAND = 0.10          # no-trade band on in-regime leverage changes
SPLIT = pd.Timestamp("2013-01-01")

s = load_close(XLSX)
dates = s.index
P = s.values[None, :]
F = features(P)
START = 199
reg, _ = regime(F, START)
D = dates[START:]
i_split = np.searchsorted(D, SPLIT)


def score(E, X):
    E = E.ravel()
    m = metrics(E)
    a, b = E[:i_split + 1], E[i_split:] / E[i_split]
    m1, m2 = metrics(a), metrics(b)
    inm = X[0, START:] > 0
    m.update(CAGR_00_12=m1["CAGR"], DD_00_12=m1["MaxDD"], Ulc_00_12=m1["Ulcer"],
             CAGR_13_26=m2["CAGR"], DD_13_26=m2["MaxDD"], Ulc_13_26=m2["Ulcer"],
             AvgL=X[0, START:][inm].mean(),
             Turnover=np.abs(np.diff(X[0, START:])).sum() / (len(E) / 252))
    return m


rows, curves = [], {}


def add(name, family, Lt, band=BAND):
    E, X = run(F, reg, Lt, START, band=band)
    m = score(E, X)
    rows.append(dict(name=name, family=family, **m))
    curves[name] = E.ravel()


# ---- static frontier
for L in np.round(np.arange(0.5, 3.01, 0.05), 2):
    add(f"static {L:.2f}", "static", pol.static(F, L), band=0.0)

# ---- dip ladders (the "add at -5% / -10%" idea), sticky & level, ATH & 52w anchors
bases = [1.1, 1.2, 1.3, 1.4]
ladders = {
    "+0.2@10": lambda b: [(0.10, b + 0.2)],
    "+0.1@5,+0.2@10": lambda b: [(0.05, b + 0.1), (0.10, b + 0.2)],
    "+0.2@5,+0.4@10": lambda b: [(0.05, b + 0.2), (0.10, b + 0.4)],
    "+0.2@10,+0.4@20": lambda b: [(0.10, b + 0.2), (0.20, b + 0.4)],
    "+0.3@10,+0.6@20": lambda b: [(0.10, b + 0.3), (0.20, b + 0.6)],
    "+0.1@5,+0.2@10,+0.3@15": lambda b: [(0.05, b + 0.1), (0.10, b + 0.2), (0.15, b + 0.3)],
    "+0.5@15": lambda b: [(0.15, b + 0.5)],
    "+0.4@20": lambda b: [(0.20, b + 0.4)],
}
for b, (lname, fn), anchor, sticky in itertools.product(bases, ladders.items(), ["ath", "52w"], [True, False]):
    for reset in ([0.0, 0.02] if sticky else [0.0]):
        nm = f"ladder b{b} {lname} {anchor} {'sticky' if sticky else 'level'}" + (f" r{reset}" if sticky else "")
        add(nm, f"ladder-{'sticky' if sticky else 'level'}", pol.ladder(F, b, fn(b), anchor, sticky, reset))

# ---- de-lever at highs / re-lever on dips (lower leverage at the high)
for hi, mid, lo in [(1.1, 1.4, 1.7), (1.2, 1.4, 1.6), (1.2, 1.5, 1.8), (1.0, 1.4, 1.8), (1.2, 1.3, 1.5)]:
    for anchor in ["ath", "52w"]:
        add(f"highs-delever {hi}/{mid}@5/{lo}@10 {anchor}", "delever-highs",
            pol.ladder(F, hi, [(0.05, mid), (0.10, lo)], anchor, True, 0.0))

# ---- volatility targeting
for tgt, hl, lmin, lmax in itertools.product([0.20, 0.24, 0.28, 0.32, 0.36, 0.40],
                                             [10, 20, 40], [0.5, 1.0], [1.6, 2.0, 2.5]):
    add(f"voltarget {tgt:.2f} hl{hl} [{lmin},{lmax}]", "voltarget", pol.voltarget(F, tgt, hl, lmin, lmax))

# ---- inverse variance (Kelly-style)
for lref, vref, hl, lmin, lmax in itertools.product([1.2, 1.4, 1.6], [0.20], [20, 40], [0.5, 1.0], [2.0, 2.5]):
    add(f"invvar {lref}@{vref} hl{hl} [{lmin},{lmax}]", "invvar", pol.invvar(F, lref, vref, hl, lmin, lmax))

# ---- vol gate (two-level)
for Lhi, Llo, vcut in itertools.product([1.5, 1.7, 2.0], [1.0, 1.2], [0.18, 0.22, 0.26, 0.30]):
    add(f"volgate {Lhi}/{Llo} cut{vcut}", "volgate", pol.vol_gate(F, Lhi, Llo, vcut, 20))

# ---- distance from 200-SMA
for ln, lf in [(1.6, 1.1), (1.8, 1.0), (1.5, 1.2), (1.1, 1.6), (2.0, 1.2), (1.7, 1.3)]:
    add(f"smadist near{ln} far{lf}", "sma-dist", pol.sma_dist(F, ln, lf, 0.20))

# ---- short-term oversold boost
for base, boost, thr, hold in itertools.product([1.2, 1.3, 1.4], [0.3, 0.6], [5, 10, 20], [3, 5, 10]):
    add(f"oversold b{base}+{boost} rsi2<{thr} {hold}d", "oversold", pol.oversold(F, base, boost, thr, hold))

# ---- regime re-entry boost
for base, bl, days in itertools.product([1.2, 1.3, 1.4], [1.6, 2.0], [63, 126, 252]):
    add(f"entryboost b{base} {bl} {days}d", "entry-boost", pol.entry_boost(F, reg, base, bl, days))

# ---- dip ladder only while calm
for base, vmax in itertools.product([1.2, 1.3, 1.4], [0.22, 0.26, 0.30]):
    add(f"dipcalm b{base} +0.2@5,+0.4@10 v<{vmax}", "dip-calm",
        pol.dip_calm(F, base, [(0.05, base + 0.2), (0.10, base + 0.4)], vmax))

# ---- vol target + dip add-on
for tgt, steps_name, steps in itertools.product([0.24, 0.28, 0.32],
                                                ["+0.2@10", "+0.2@5,+0.4@10"],
                                                [[(0.10, 0.2)], [(0.05, 0.2), (0.10, 0.4)]]):
    if (steps_name == "+0.2@10") != (len(steps) == 1):
        continue
    add(f"vt+dip {tgt:.2f} {steps_name}", "vt+dip", pol.vt_ladder(F, tgt, steps, 20, 1.0, 2.0))

R = pd.DataFrame(rows).set_index("name")

# ---------------- matched-risk comparison against the static frontier
st = R[R.family == "static"].copy()
st["L"] = [float(n.split()[1]) for n in st.index]


def matched(col_risk, col_ret="CAGR", frame=st):
    x = -frame[col_risk].values if "DD" in col_risk else frame[col_risk].values
    o = np.argsort(x)
    return lambda v: np.interp(-v if "DD" in col_risk else v, x[o], frame[col_ret].values[o])


f_dd, f_ulc, f_vol = matched("MaxDD"), matched("Ulcer"), matched("Vol")
f_dd1, f_dd2 = matched("DD_00_12", "CAGR_00_12"), matched("DD_13_26", "CAGR_13_26")
f_u1, f_u2 = matched("Ulc_00_12", "CAGR_00_12"), matched("Ulc_13_26", "CAGR_13_26")
R["xs_vs_DD"] = R.CAGR - f_dd(R.MaxDD)
R["xs_vs_Ulcer"] = R.CAGR - f_ulc(R.Ulcer)
R["xs_vs_Vol"] = R.CAGR - f_vol(R.Vol)
R["xs_Ulc_00_12"] = R.CAGR_00_12 - f_u1(R.Ulc_00_12)
R["xs_Ulc_13_26"] = R.CAGR_13_26 - f_u2(R.Ulc_13_26)
R["xs_DD_00_12"] = R.CAGR_00_12 - f_dd1(R.DD_00_12)
R["xs_DD_13_26"] = R.CAGR_13_26 - f_dd2(R.DD_13_26)
R["static_L_same_Ulcer"] = np.interp(R.Ulcer, st.Ulcer.values[np.argsort(st.Ulcer.values)],
                                     st.L.values[np.argsort(st.Ulcer.values)])
R["robust"] = R[["xs_vs_DD", "xs_vs_Ulcer", "xs_vs_Vol", "xs_Ulc_00_12", "xs_Ulc_13_26"]].min(axis=1)

R.to_csv(os.path.join(OUT, "hist_results.csv"))
np.savez_compressed(os.path.join(OUT, "hist_curves.npz"), dates=D.values.astype("datetime64[D]"),
                    **{k.replace(" ", "_").replace("/", "-"): v for k, v in curves.items()})

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
cols = ["CAGR", "MaxDD", "Ulcer", "Vol", "Sharpe", "AvgL", "Turnover", "xs_vs_DD", "xs_vs_Ulcer",
        "xs_vs_Vol", "xs_Ulc_00_12", "xs_Ulc_13_26", "robust", "static_L_same_Ulcer"]
fmt = R[cols].copy()
print("Static reference:")
print(fmt.loc[[f"static {x:.2f}" for x in [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]]].round(3).to_string())
print("\nFamily summary (share of variants beating the static frontier at matched Ulcer, both halves):")
dyn = R[R.family != "static"]
summ = dyn.groupby("family").agg(
    n=("CAGR", "size"),
    med_xs_Ulcer=("xs_vs_Ulcer", "median"),
    best_xs_Ulcer=("xs_vs_Ulcer", "max"),
    share_beat_both_halves=("robust", lambda x: (x > 0).mean()),
    med_xs_DD=("xs_vs_DD", "median"))
print(summ.round(4).sort_values("med_xs_Ulcer", ascending=False).to_string())
print("\nTop 30 by robust score (min excess across DD/Ulcer/Vol and both halves):")
print(fmt.sort_values("robust", ascending=False).head(30).round(3).to_string())
print("\nYour idea — ladders from base 1.3:")
print(fmt[fmt.index.str.startswith("ladder b1.3")].sort_values("xs_vs_Ulcer", ascending=False).round(3).to_string())
