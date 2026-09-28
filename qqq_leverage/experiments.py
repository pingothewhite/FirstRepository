"""Follow-up experiments behind the recommendation (all run through scorecard.py's
9 tests: history full / 3 eras / 1-day-late execution, plus 4 Monte Carlo generators).

    python experiments.py ladders      # your question: add at -5/-10/-15/-20/-25%?
    python experiments.py oversold     # RSI(2) / short-term drop sensitivity
    python experiments.py anchors      # drawdown measured from ATH vs 52w high vs regime peak
    python experiments.py final        # the final candidates (writes final_recommended_*.csv)

Set QQQ_XLSX to the full 1999+ export first."""
import itertools, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorecard as sc
import policies as pol
from lev_engine import load_close

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def statics(F):
    return {f"static {L:.1f}": (pol.static(F, L), 0.0) for L in sc.STATIC_GRID}


def build_ladders(F, reg, dates=None):
    C = statics(F)
    for y, x in itertools.product([0.05, 0.10, 0.15, 0.20, 0.25], [0.2, 0.4, 0.7]):
        C[f"1.3 +{x} at -{int(y*100)}% (reset at ATH)"] = (pol.ladder(F, 1.3, [(y, 1.3 + x)], "ath", True, 0.0), 0.1)
        C[f"1.3 +{x} at -{int(y*100)}% (level)"] = (pol.ladder(F, 1.3, [(y, 1.3 + x)], "ath", False), 0.1)
    for hi in [1.0, 1.1, 1.2]:
        C[f"highs {hi}, 1.3 at -5%, 1.5 at -10%"] = (pol.ladder(F, hi, [(0.05, 1.3), (0.10, 1.5)], "ath", True, 0.0), 0.1)
    return C


def build_oversold(F, reg, dates=None):
    C = statics(F)
    for thr in [2, 3, 5, 7, 10]:
        C[f"RSI2<{thr} 1.3+0.6 5d"] = (pol.oversold(F, 1.3, 0.6, thr, 5), 0.1)
    for hold in [2, 3, 8, 12]:
        C[f"RSI2<5 1.3+0.6 {hold}d"] = (pol.oversold(F, 1.3, 0.6, 5, hold), 0.1)
    for d, dr in [(2, 0.04), (3, 0.05), (3, 0.07), (5, 0.06), (5, 0.08)]:
        C[f"{d}d drop>={int(dr*100)}% 1.3+0.6 5d"] = (pol.drop_boost(F, 1.3, 0.6, d, dr, 5), 0.1)
    for tgt, b in itertools.product([0.28, 0.32, 0.36], [0.2, 0.4, 0.6]):
        C[f"VT{int(tgt*100)} [1,1.6] +RSI2 {b}"] = (pol.vt_os(F, tgt, 10, 1.0, 1.6, b), 0.1)
    return C


def build_anchors(F, reg, dates=None):
    pol.add_regime_drawdown(F, reg)
    C = statics(F)
    vt = pol.voltarget(F, 0.32, 10, 1.0, 1.6)
    for a in ["ath", "52w", "regime"]:
        C[f"A {a}: 1.3 +0.4 @-20% +0.7 @-25%"] = (pol.deep_dip(F, 1.3, [(0.20, 0.4), (0.25, 0.7)], a), 0.1)
        C[f"A' {a}: 1.3 +0.4 @-15% +0.7 @-20%"] = (pol.deep_dip(F, 1.3, [(0.15, 0.4), (0.20, 0.7)], a), 0.1)
        C[f"C {a}: VT32 + 0.4 @-20%"] = (np.minimum(pol.deep_dip(F, vt, [(0.20, 0.4)], a), 2.0), 0.1)
    return C


def build_final(F, reg, dates=None):
    C = statics(F)
    os_ = pol.oversold(F, 0.0, 0.4, 5, 5)
    vt32 = pol.voltarget(F, 0.32, 10, 1.0, 1.6)
    vt28 = pol.voltarget(F, 0.28, 10, 1.0, 1.6)
    C["R1 VT32 + 0.4@-20% + 0.3@-25% (52w)"] = (np.minimum(pol.deep_dip(F, vt32, [(0.20, 0.4), (0.25, 0.7)], "52w"), 2.0), 0.1)
    C["R2 VT32 + 0.4@-20% (52w)"] = (np.minimum(pol.deep_dip(F, vt32, [(0.20, 0.4)], "52w"), 2.0), 0.1)
    C["R3 VT28 + 0.4@-20% + 0.3@-25% (52w)"] = (np.minimum(pol.deep_dip(F, vt28, [(0.20, 0.4), (0.25, 0.7)], "52w"), 2.0), 0.1)
    C["R4 R1 + RSI2 boost 0.4"] = (np.minimum(pol.deep_dip(F, vt32, [(0.20, 0.4), (0.25, 0.7)], "52w") + os_, 2.0), 0.1)
    C["U1 ladder 1.3->1.5@-10%"] = (pol.ladder(F, 1.3, [(0.10, 1.5)], "ath", True, 0.0), 0.1)
    C["U2 ladder 1.3->1.4@-5%->1.5@-10%"] = (pol.ladder(F, 1.3, [(0.05, 1.4), (0.10, 1.5)], "ath", True, 0.0), 0.1)
    C["DD deep dip only 1.3 +0.4@-20% +0.7@-25% (52w)"] = (pol.deep_dip(F, 1.3, [(0.20, 0.4), (0.25, 0.7)], "52w"), 0.1)
    C["VT32 only (1.0-1.6x)"] = (vt32, 0.1)
    return C


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "final"
    sc.build = {"ladders": build_ladders, "oversold": build_oversold,
                "anchors": build_anchors, "final": build_final}[which]
    s = load_close(os.environ.get("QQQ_XLSX", "QQQ-history.xlsx"))
    tests = sc.history_tests(s)
    mct, plans = sc.mc_tests()
    tests.update(mct)
    xs = pd.DataFrame({k: sc.frontier_excess(v) for k, v in tests.items()})
    dyn = xs[~xs.index.str.startswith("static")].copy()
    dyn["worst"] = dyn.min(axis=1)
    dyn["won"] = (xs.loc[dyn.index] > 0).sum(axis=1)
    pd.set_option("display.width", 280)
    print("Excess CAGR vs static leverage at the same risk (Ulcer), %/yr:")
    print((dyn.drop(columns="won") * 100).round(2).assign(won=dyn.won).to_string())
    tag = {"final": "final_recommended"}.get(which, which)
    dyn.to_csv(os.path.join(OUT, f"{tag}_scorecard.csv"))
    pd.concat(tests, names=["test"]).to_csv(os.path.join(OUT, f"{tag}_tests.csv"))
    pd.concat(plans, names=["gen"]).to_csv(os.path.join(OUT, f"{tag}_plans_nomargin.csv"))
