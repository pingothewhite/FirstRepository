# ============================================================
#  QQQ LEVERAGED PLAN — INTERACTIVE MONTE CARLO DASHBOARD  (v6)
#  Paste this entire script into ONE Google Colab cell.
#
#  NEW IN v6 — DYNAMIC LEVERAGE
#   `leverage_mode` picks how much leverage to hold while the 200-SMA
#   band regime is ON (the regime / SGOV logic is unchanged):
#     "static"                   v5 behaviour: `leverage` all the time.
#     "vol_target_plus_deep_dip" RECOMMENDED. leverage = vol_target ÷
#                                QQQ's recent volatility, kept between
#                                lev_min and lev_max, PLUS `dip_add`
#                                while QQQ is >= `dip_trigger` below
#                                its 52-week high (+`dip_add_2` more at
#                                `dip_trigger_2`), removed as soon as
#                                it recovers above the trigger.
#                                Defaults: 32% target, 1.0-1.6x,
#                                +0.4x at -20%, +0.3x more at -25%.
#     "vol_target"               volatility scaling only.
#     "deep_dip"                 `leverage` + the deep-dip add-on.
#     "drawdown_ladder"          your step-up idea: e.g. 1.4 at -5%,
#                                1.5 at -10%, back to `leverage` at a
#                                new all-time high.
#   Optional RSI(2) boost for any mode: +rsi2_boost for rsi2_hold_days
#   after QQQ's 2-day RSI closes below rsi2_threshold.
#   compare_to_static = True runs the SAME random paths at static
#   `leverage` and prints a side-by-side table + comparison chart.
#   Engine is vectorised across paths (much faster than v5). In static
#   mode with trade_cost_bps = 0 it reproduces v5 to the cent.
#
#  v5: 2x2 charts, log + linear equity curves, P25/P75 columns,
#      surplus income invested.  v3: margin & secure outside debt
#      waterfall.
#
#  NOT FINANCIAL ADVICE. A statistical model of one strategy.
# ============================================================

# ═══════════ SECTION 1 — PORTFOLIO ═══════════
current_balance       = 760000   #@param {type:"number"}
goal_target           = 1500000  #@param {type:"number"}

# ═══════════ SECTION 2 — STRATEGY SIGNAL ═══════════
leverage              = 1.3      #@param {type:"slider", min:1.0, max:2.0, step:0.1}
band_pct_up           = 0.04     #@param {type:"slider", min:0.0, max:0.15, step:0.01}
band_pct_down         = 0.08     #@param {type:"slider", min:0.0, max:0.15, step:0.01}
entry_days            = 3        #@param {type:"slider", min:1, max:20, step:1}
exit_days             = 15       #@param {type:"slider", min:1, max:40, step:1}
sgov_annual_yield     = 0.042    #@param {type:"number"}

# ═══════════ SECTION 2b — DYNAMIC LEVERAGE (NEW IN v6) ═══════════
leverage_mode         = "vol_target_plus_deep_dip"  #@param ["static", "vol_target_plus_deep_dip", "vol_target", "deep_dip", "drawdown_ladder"]
vol_target            = 0.32     #@param {type:"slider", min:0.20, max:0.45, step:0.01}
vol_halflife_days     = 10       #@param {type:"slider", min:5, max:40, step:1}
lev_min               = 1.0      #@param {type:"slider", min:0.5, max:1.5, step:0.1}
lev_max               = 1.6      #@param {type:"slider", min:1.0, max:2.0, step:0.1}
dip_trigger           = 0.20     #@param {type:"slider", min:0.05, max:0.35, step:0.01}
dip_add               = 0.4      #@param {type:"slider", min:0.0, max:1.0, step:0.1}
dip_trigger_2         = 0.25     #@param {type:"slider", min:0.05, max:0.40, step:0.01}
dip_add_2             = 0.3      #@param {type:"slider", min:0.0, max:1.0, step:0.1}
dip_anchor            = "52w"    #@param ["52w", "ath"]
#   drawdown_ladder mode: "drawdown:leverage" pairs; resets to `leverage` at a new ATH
ladder_levels         = "0.05:1.4, 0.10:1.5"  #@param {type:"string"}
rsi2_boost            = 0.0      #@param {type:"slider", min:0.0, max:0.8, step:0.1}
rsi2_threshold        = 5        #@param {type:"slider", min:2, max:15, step:1}
rsi2_hold_days        = 5        #@param {type:"slider", min:1, max:10, step:1}
lev_cap               = 2.0      #@param {type:"slider", min:1.0, max:2.0, step:0.1}
rebalance_band        = 0.10     #@param {type:"slider", min:0.0, max:0.30, step:0.05}
trade_cost_bps        = 5        #@param {type:"number"}
compare_to_static     = True     #@param {type:"boolean"}

# ═══════════ SECTION 3 — SPENDING ═══════════
monthly_floor         = 6500     #@param {type:"number"}
monthly_pct           = 0.007    #@param {type:"slider", min:0.0, max:0.02, step:0.0005}

# ═══════════ SECTION 4 — WORK / INCOME ═══════════
monthly_income        = 4000     #@param {type:"number"}
#   "never" | "immediately" | "if_portfolio_falls_to"
work_starts           = "never"  #@param ["never", "immediately", "if_portfolio_falls_to"]
work_trigger_balance  = 500000   #@param {type:"number"}
#   "at_target_balance" | "after_fixed_years" | "never_stop"
work_stops            = "after_fixed_years"  #@param ["at_target_balance", "after_fixed_years", "never_stop"]
work_stop_target      = 1500000  #@param {type:"number"}
work_max_years        = 5        #@param {type:"slider", min:1, max:30, step:1}
work_can_restart      = True     #@param {type:"boolean"}

# ═══════════ SECTION 5 — MARGIN & SECURE OUTSIDE DEBT ═══════════
use_margin                 = True     #@param {type:"boolean"}
margin_dd_trigger          = 0.20     #@param {type:"slider", min:0.05, max:0.50, step:0.05}
repay_within_pct           = 0.02     #@param {type:"slider", min:0.0, max:0.10, step:0.01}
max_margin_ratio           = 0.35     #@param {type:"slider", min:0.05, max:0.60, step:0.05}
margin_rate_annual         = 0.058    #@param {type:"number"}
use_outside_debt           = True     #@param {type:"boolean"}
outside_debt_limit         = 140000   #@param {type:"number"}
outside_debt_rate_annual   = 0.12     #@param {type:"number"}
maint_qld                  = 0.50     #@param {type:"number"}
maint_qqq                  = 0.30     #@param {type:"number"}
maint_sgov                 = 0.04     #@param {type:"number"}

# ═══════════ SECTION 6 — SIMULATION ═══════════
horizon_years         = 10       #@param {type:"slider", min:5, max:40, step:1}
n_paths               = 2000     #@param {type:"slider", min:500, max:5000, step:500}
n_plot_paths          = 150      #@param {type:"slider", min:20, max:400, step:10}
random_seed           = 42       #@param {type:"number"}

# ═══════════ SECTION 7 — CHART OPTIONS ═══════════
linear_chart_cap_pct  = 90       #@param {type:"slider", min:75, max:100, step:5}
linear_chart_headroom = 1.15     #@param {type:"number"}
# ============================================================

import glob, os, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
try:
    from IPython.display import display
except ImportError:                      # plain Python fallback
    def display(x):
        print(x.data if hasattr(x, "data") else x)

# ============================================================
# 1. LOAD DATA  (yfinance, else a local/uploaded QQQ export)
# ============================================================
def _load_file(path):
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return df.set_index("Date").sort_index()


def load_qqq():
    local = os.environ.get("QQQ_FILE", "")
    if local and os.path.exists(local):
        print(f"Loading {local} ...")
        return _load_file(local)
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yfinance"], check=False)
        import yfinance as yf
        print("Downloading QQQ history...")
        q = yf.download("QQQ", start="1999-03-10", auto_adjust=False, progress=False)
        if isinstance(q.columns, pd.MultiIndex):
            q.columns = q.columns.get_level_values(0)
        q = q.dropna(subset=["Close"])
        if len(q) > 1000:
            return q
    except Exception as e:
        print(f"yfinance failed ({e.__class__.__name__}).")
    hits = sorted(glob.glob("*QQQ*.xlsx") + glob.glob("*QQQ*.csv") + glob.glob("/content/*QQQ*"))
    if hits:
        print(f"Loading {hits[0]} ...")
        return _load_file(hits[0])
    from google.colab import files       # last resort: ask for the export
    up = files.upload()
    return _load_file(next(iter(up)))


qqq = load_qqq()
close = qqq["Close"].values.astype(float)
dates = pd.DatetimeIndex(qqq.index)
sma_series = pd.Series(close).rolling(200).mean().values
last_close, last_sma, last_date = close[-1], sma_series[-1], dates[-1]

# ============================================================
# 2. CURRENT SIGNAL STATE
# ============================================================
def build_regime_band(c, s):
    upper, lower = s * (1 + band_pct_up), s * (1 - band_pct_down)
    state = 1 if c[199] > s[199] else 0
    above_cnt = below_cnt = 0
    for i in range(199, len(c)):
        if c[i] > upper[i]:   above_cnt += 1; below_cnt = 0
        elif c[i] < lower[i]: below_cnt += 1; above_cnt = 0
        if state == 0 and above_cnt >= entry_days:
            state = 1; above_cnt = below_cnt = 0
        elif state == 1 and below_cnt >= exit_days:
            state = 0; above_cnt = below_cnt = 0
    return state, above_cnt, below_cnt


current_regime, cur_above, cur_below = build_regime_band(close, sma_series)
upper_now, lower_now = last_sma * (1 + band_pct_up), last_sma * (1 - band_pct_down)

# --- dynamic-leverage indicators, carried forward from history into each path
VOL_LAM = 0.5 ** (1 / vol_halflife_days)
hist_r = np.r_[0.0, close[1:] / close[:-1] - 1]
_v = np.mean(hist_r[1:22] ** 2)
for x in hist_r:
    _v = VOL_LAM * _v + (1 - VOL_LAM) * x * x
hist_var = _v
_u = _d = 0.0
for x in hist_r:
    _u = 0.5 * _u + 0.5 * max(x, 0.0)
    _d = 0.5 * _d + 0.5 * max(-x, 0.0)
hist_rsi_u, hist_rsi_d = _u, _d
hist_ath = close.max()
hi52_now = close[-252:].max()


def parse_ladder(txt):
    out = []
    for part in txt.split(","):
        if ":" in part:
            a, b = part.split(":")
            out.append((abs(float(a)), float(b)))
    return sorted(out)


LADDER = parse_ladder(ladder_levels)


def target_leverage(vol_ann, dd_ath, dd_52, rsi2_cnt, ladder_lvl, mode):
    """In-regime target leverage (arrays or scalars)."""
    dd = dd_ath if dip_anchor == "ath" else dd_52
    dip = np.where(dd <= -dip_trigger, dip_add, 0.0) + np.where(dd <= -dip_trigger_2, dip_add_2, 0.0)
    vt = np.clip(vol_target / np.maximum(vol_ann, 1e-6), lev_min, lev_max)
    if mode == "static":
        L = np.full(np.shape(vol_ann), float(leverage))
    elif mode == "vol_target":
        L = vt
    elif mode == "vol_target_plus_deep_dip":
        L = vt + dip
    elif mode == "deep_dip":
        L = leverage + dip
    elif mode == "drawdown_ladder":
        L = ladder_lvl
    else:
        raise ValueError(mode)
    if mode != "static":
        L = L + np.where(rsi2_cnt > 0, rsi2_boost, 0.0)
        L = np.minimum(L, lev_cap)
    return L


vol_now = np.sqrt(252 * hist_var)
dd_ath_now, dd_52_now = last_close / hist_ath - 1, last_close / hi52_now - 1
rsi2_now = 100 * hist_rsi_u / max(hist_rsi_u + hist_rsi_d, 1e-12)
lad_now = leverage
for d_, l_ in LADDER:
    if dd_ath_now <= -d_:
        lad_now = l_
dyn_now = float(target_leverage(np.array(vol_now), np.array(dd_ath_now), np.array(dd_52_now),
                                np.array(1.0 if rsi2_now < rsi2_threshold else 0.0), lad_now, leverage_mode))

print("=" * 70)
print(f"CURRENT SIGNAL STATE  (as of {last_date.date()})")
print("=" * 70)
print(f"QQQ close: ${last_close:,.2f} | 200-SMA: ${last_sma:,.2f} "
      f"({(last_close/last_sma-1)*100:+.1f}%)")
print(f"Band: ${lower_now:,.2f} - ${upper_now:,.2f}")
print(f"REGIME: {'LONG' if current_regime==1 else 'SGOV (defensive)'}"
      f" | above={cur_above}/{entry_days}, below={cur_below}/{exit_days}")
print(f"Realised vol ({vol_halflife_days}d half-life): {vol_now*100:.1f}% | "
      f"from ATH: {dd_ath_now*100:+.1f}% | from 52w high: {dd_52_now*100:+.1f}% | RSI(2): {rsi2_now:.0f}")
if current_regime == 1:
    print(f"LEVERAGE NOW: {leverage_mode} -> {dyn_now:.2f}x   (static setting: {leverage:.2f}x)")

# --- plain-English rules ---
print("\n" + "-" * 70)
mode_desc = {
    "static": f"Hold {leverage}x whenever the regime is ON.",
    "vol_target": f"Leverage = {vol_target*100:.0f}% / QQQ realised vol, kept within {lev_min}-{lev_max}x.",
    "vol_target_plus_deep_dip": (f"Leverage = {vol_target*100:.0f}% / QQQ realised vol, kept within {lev_min}-{lev_max}x, "
                                 f"plus {dip_add}x while QQQ is >= {dip_trigger*100:.0f}% below its "
                                 f"{'all-time' if dip_anchor=='ath' else '52-week'} high"
                                 + (f" (+{dip_add_2}x more at -{dip_trigger_2*100:.0f}%)" if dip_add_2 > 0 else "") + "."),
    "deep_dip": (f"Hold {leverage}x, plus {dip_add}x while QQQ is >= {dip_trigger*100:.0f}% below its "
                 f"{'all-time' if dip_anchor=='ath' else '52-week'} high."),
    "drawdown_ladder": (f"Hold {leverage}x at highs; step up to " +
                        ", ".join(f"{l}x at -{d*100:.0f}%" for d, l in LADDER) + "; back to base at a new ATH."),
}[leverage_mode]
if leverage_mode != "static" and rsi2_boost > 0:
    mode_desc += f" +{rsi2_boost}x for {rsi2_hold_days}d after RSI(2) < {rsi2_threshold}."
print("LEVERAGE RULE: " + mode_desc)
if monthly_income <= 0 or work_starts == "never":
    work_desc = "No job. All withdrawals come from the portfolio."
else:
    start_txt = ("You start working on day 1" if work_starts == "immediately"
                 else f"You return to work only if the portfolio falls to ${work_trigger_balance:,.0f}")
    stop_txt = (f"and stop once it recovers to ${work_stop_target:,.0f}" if work_stops == "at_target_balance"
                else f"and stop after {work_max_years} years regardless" if work_stops == "after_fixed_years"
                else "and never stop once started")
    restart_txt = ""
    if work_starts == "if_portfolio_falls_to" and work_stops != "never_stop":
        restart_txt = (" You can be called back if it falls again." if work_can_restart
                       else " Once you stop, you never work again.")
    work_desc = f"{start_txt}, earning ${monthly_income:,.0f}/mo, {stop_txt}.{restart_txt}"
print("WORK RULE:     " + work_desc)
if use_margin:
    margin_desc = (f"Within {margin_dd_trigger*100:.0f}% of your high-water mark you SELL shares. "
                   f"Deeper than that (and still invested) you BORROW on margin instead, "
                   f"capped at {max_margin_ratio*100:.0f}% of assets. "
                   f"All debt is repaid when QQQ is within {repay_within_pct*100:.0f}% of its own ATH.")
    if use_outside_debt:
        margin_desc += (f" ${outside_debt_limit:,.0f} of secure outside debt is injected only to "
                        f"cure a margin shortfall.")
else:
    margin_desc = "No borrowing. Every withdrawal sells shares, even at the bottom."
print("CASH RULE:     " + margin_desc)
print("-" * 70 + "\n")

# ============================================================
# 3. RANDOM PATHS  (identical generator to v5)
# ============================================================
rets = pd.Series(close).pct_change().values
post05 = rets[np.array(dates >= pd.Timestamp("2005-01-01"))]
post05 = post05[~np.isnan(post05)]
bearpool = rets[np.array((dates >= "2021-11-01") & (dates <= "2022-12-31"))]
bearpool = bearpool[~np.isnan(bearpool)]
seed_closes = close[-199:]

QLD_DRAG, QQQ_DRAG = 0.0545 / 252, 0.002 / 252
SGOV_D    = sgov_annual_yield / 252
MARG_D    = margin_rate_annual / 252
OUTSIDE_D = outside_debt_rate_annual / 252
BLOCK     = 21
TC        = trade_cost_bps / 1e4
nd = int(252 * horizon_years)
nb = nd // BLOCK + 2
work_days_cap = int(252 * work_max_years)

rng = np.random.default_rng(random_seed)
R = np.empty((n_paths, nd))
for p in range(n_paths):
    st = rng.integers(0, len(post05) - BLOCK, nb)
    r = np.concatenate([post05[s:s + BLOCK] for s in st])[:nd]
    if rng.random() < 0.25 and len(bearpool) > BLOCK * 2:
        bs = rng.integers(0, len(bearpool) - BLOCK, 7)
        r[:126] = np.concatenate([bearpool[s:s + BLOCK] for s in bs])[:126]
    R[p] = r
PR = last_close * np.cumprod(1 + R, axis=1)                        # path prices
ALLC = np.concatenate([np.tile(np.r_[seed_closes, last_close], (n_paths, 1)), PR], axis=1)
CS = np.cumsum(ALLC, axis=1)
SMA = (CS[:, 200:] - CS[:, :-200]) / 200                           # == v5's convolve()[1:]

# --- indicator paths for the dynamic rules (all known at each close)
TAIL = np.tile(close[-251:], (n_paths, 1))
HI52 = pd.DataFrame(np.concatenate([TAIL, PR], axis=1).T).rolling(252).max().values.T[:, 251:]
ATH = np.maximum(hist_ath, np.maximum.accumulate(PR, axis=1))
DD_ATH, DD_52 = PR / ATH - 1, PR / HI52 - 1
VOL = np.empty((n_paths, nd))
RSICNT = np.zeros((n_paths, nd))
LAD = np.empty((n_paths, nd))
v = np.full(n_paths, hist_var)
u = np.full(n_paths, hist_rsi_u)
dn = np.full(n_paths, hist_rsi_d)
cnt = np.zeros(n_paths)
lad = np.full(n_paths, float(lad_now))
for i in range(nd):
    x = R[:, i]
    v = VOL_LAM * v + (1 - VOL_LAM) * x * x
    VOL[:, i] = np.sqrt(252 * v)
    u = 0.5 * u + 0.5 * np.maximum(x, 0)
    dn = 0.5 * dn + 0.5 * np.maximum(-x, 0)
    rsi = 100 * u / np.maximum(u + dn, 1e-12)
    cnt = np.where(rsi < rsi2_threshold, rsi2_hold_days, np.maximum(cnt - 1, 0))
    RSICNT[:, i] = cnt
    lvl = np.full(n_paths, float(leverage))
    for d_, l_ in LADDER:
        lvl = np.where(DD_ATH[:, i] <= -d_, l_, lvl)
    lad = np.where(DD_ATH[:, i] >= 0, float(leverage), np.maximum(lad, lvl))
    LAD[:, i] = lad


# ============================================================
# 4. MONTE CARLO ENGINE  (vectorised across paths)
# ============================================================
def run_engine(mode):
    TGT = target_leverage(VOL, DD_ATH, DD_52, RSICNT, LAD, mode)
    n = n_paths
    A = np.full(n, float(current_balance)); D = np.zeros(n); H = np.zeros(n)
    hwm = A.copy(); ihw = np.full(n, last_close)
    state = np.full(n, current_regime); a_cnt = np.full(n, cur_above); b_cnt = np.full(n, cur_below)
    L0 = dyn_now if (mode != "static") else float(leverage)
    expo = np.where(state == 1, L0, 0.0)

    working = np.full(n, monthly_income > 0 and work_starts == "immediately")
    ever_worked = working.copy(); work_finished = np.zeros(n, bool)
    wdays = np.zeros(n); spells = working.astype(float)
    t_work = np.where(working, 0.0, np.nan)
    bal_yearly = np.full((n, horizon_years), np.nan)
    bal_plot = np.full((n_plot_paths, nd + 1), np.nan)
    bal_plot[:, 0] = (A - D - H)[:n_plot_paths]
    t_goal = np.full(n, np.nan)
    work_income = np.zeros(n); contrib = np.zeros(n)
    peak_m = np.zeros(n); peak_o = np.zeros(n)
    mcall = np.zeros(n, bool); interest = np.zeros(n)
    eq_peak = A - D - H; maxdd = np.zeros(n)
    lev_sum = np.zeros(n); lev_days = np.zeros(n)

    for i in range(nd):
        E = A - D - H
        al = E > 0
        # ---- market (exposure chosen at the previous close)
        L = expo
        g = np.where(L <= 0, 1 + SGOV_D,
             np.where(L >= 1, 1 + L * R[:, i] - (L - 1) * QLD_DRAG - np.maximum(2 - L, 0) * QQQ_DRAG,
                      1 + L * R[:, i] + (1 - L) * SGOV_D - L * QQQ_DRAG))
        A = np.where(al, A * g, A)
        D = np.where(al, D * (1 + MARG_D), D)
        H = np.where(al, H * (1 + OUTSIDE_D), H)
        interest += np.where(al, D * MARG_D + H * OUTSIDE_D, 0.0)
        lev_sum += np.where(al & (L > 0), L, 0.0); lev_days += (al & (L > 0))

        # ---- regime signal
        p_i = PR[:, i]
        up_b, lo_b = SMA[:, i] * (1 + band_pct_up), SMA[:, i] * (1 - band_pct_down)
        above, below = p_i > up_b, p_i < lo_b
        a_new = np.where(above, a_cnt + 1, np.where(below, 0, a_cnt))
        b_new = np.where(below, b_cnt + 1, np.where(above, 0, b_cnt))
        go_in = (state == 0) & (a_new >= entry_days)
        go_out = (state == 1) & (b_new >= exit_days)
        s_new = np.where(go_in, 1, np.where(go_out, 0, state))
        rs = go_in | go_out
        a_new = np.where(rs, 0, a_new); b_new = np.where(rs, 0, b_new)
        state = np.where(al, s_new, state); a_cnt = np.where(al, a_new, a_cnt); b_cnt = np.where(al, b_new, b_cnt)

        # ---- leverage for tomorrow (no-trade band; entries/exits always trade)
        tgt = np.where(state == 1, TGT[:, i], 0.0)
        chg = (tgt == 0) | (expo == 0) | (np.abs(tgt - expo) >= rebalance_band - 1e-12)
        new_expo = np.where(al & chg, tgt, expo)
        A = np.where(al, A * (1 - TC * np.abs(new_expo - expo)), A)
        expo = new_expo

        E = A - D - H
        hwm = np.where(al & (E > hwm), E, hwm)
        ihw = np.where(al & (p_i > ihw), p_i, ihw)

        # ---- margin health (maintenance of the current QLD/QQQ mix)
        Lm = np.clip(expo, 1.0, 2.0)
        maint_in = (Lm - 1) * maint_qld + (2 - Lm) * maint_qqq
        maint_rate = np.where(state == 1, maint_in, maint_sgov)
        excess = (A - D) - A * maint_rate
        if use_margin and use_outside_debt:
            m = al & (D > 0) & (excess < 0.05 * A) & (H < outside_debt_limit)
            need = np.minimum(np.minimum(outside_debt_limit - H, D), 0.05 * A - excess)
            m = m & (need > 0)
            D = np.where(m, D - need, D); H = np.where(m, H + need, H)
            excess = np.where(m, (A - D) - A * maint_rate, excess)
        call = al & (D > 0) & (excess < 0)
        mcall |= call
        sell = np.minimum(A, D)
        A = np.where(call, A - sell, A); D = np.where(call, D - sell, D)

        # ---- repay all debt when the index is back near its ATH
        if use_margin:
            m = al & ((D + H) > 0) & (p_i >= ihw * (1 - repay_within_pct))
            pay = np.minimum(D + H, A * 0.9)
            use_d = np.minimum(D, pay)
            A = np.where(m, A - pay, A); D = np.where(m, D - use_d, D); H = np.where(m, H - (pay - use_d), H)

        # ---- work state machine
        E = A - D - H
        if monthly_income > 0 and work_starts != "never":
            was = working.copy()
            start = al & ~was & ((~work_finished) | work_can_restart) & \
                (work_starts == "if_portfolio_falls_to") & (E <= work_trigger_balance)
            first = start & ~ever_worked
            t_work = np.where(first, (i + 1) / 252, t_work)
            working = working | start; spells += start; ever_worked |= start
            if work_stops == "at_target_balance":
                stop = al & was & (E >= work_stop_target)
            elif work_stops == "after_fixed_years":
                stop = al & was & (wdays >= work_days_cap)
            else:
                stop = np.zeros(n, bool)
            working = working & ~stop; work_finished |= stop
            wdays += (al & working)

        # ---- monthly withdrawal
        if (i + 1) % 21 == 0:
            E = A - D - H
            need = np.maximum(monthly_floor, monthly_pct * E)
            inc = np.where(working, monthly_income, 0.0)
            work_income += np.where(al, inc, 0.0)
            draw = np.maximum(0.0, need - inc); surplus = np.maximum(0.0, inc - need)
            borrow_ok = (use_margin & (state == 1) & (E < hwm * (1 - margin_dd_trigger))
                         & ((D + draw) <= A * max_margin_ratio))
            m_d = al & (draw > 0)
            D = np.where(m_d & borrow_ok, D + draw, D)
            A = np.where(m_d & ~borrow_ok, A - draw, A)
            m_s = al & (draw <= 0) & (surplus > 0)
            A = np.where(m_s, A + surplus, A); contrib += np.where(m_s, surplus, 0.0)
            dead = al & (A - D - H <= 0)
            A = np.where(dead, 0.0, A); D = np.where(dead, 0.0, D); H = np.where(dead, 0.0, H)

        peak_m = np.where(al, np.maximum(peak_m, D), peak_m)
        peak_o = np.where(al, np.maximum(peak_o, H), peak_o)
        E = np.maximum(A - D - H, 0.0)
        eq_peak = np.maximum(eq_peak, E)
        maxdd = np.minimum(maxdd, np.where(eq_peak > 0, E / eq_peak - 1, -1.0))
        bal_plot[:, i + 1] = E[:n_plot_paths]
        if (i + 1) % 252 == 0:
            bal_yearly[:, (i + 1) // 252 - 1] = E
        t_goal = np.where(np.isnan(t_goal) & (E >= goal_target), (i + 1) / 252, t_goal)

    return dict(bal_yearly=bal_yearly, bal_plot=bal_plot, t_goal=t_goal, t_work=t_work,
                work_years=wdays / 252, work_spells=spells, work_income=work_income, contrib=contrib,
                peak_margin=peak_m, peak_outside=peak_o, margin_call=mcall, interest=interest,
                maxdd=maxdd, avg_lev=lev_sum.sum() / max(lev_days.sum(), 1))


res = run_engine(leverage_mode)
res_static = run_engine("static") if (compare_to_static and leverage_mode != "static") else None

bal_yearly, bal_daily_plot, t_goal = res["bal_yearly"], res["bal_plot"], res["t_goal"]
t_work_arr, work_years_arr, work_spells_arr = res["t_work"], res["work_years"], res["work_spells"]
work_income_arr, contrib_arr = res["work_income"], res["contrib"]
peak_margin_arr, peak_outside_arr = res["peak_margin"], res["peak_outside"]
margin_call_arr, interest_arr = res["margin_call"], res["interest"]

# ============================================================
# 5. YEAR-BY-YEAR TABLE
# ============================================================
years = np.arange(1, horizon_years + 1)
rows = []
for yi, Y in enumerate(years):
    b = bal_yearly[:, yi]
    rows.append({
        "Year": Y,
        "P10": np.percentile(b, 10),
        "P25": np.percentile(b, 25),
        "Median value": np.median(b),
        "P75": np.percentile(b, 75),
        "P90": np.percentile(b, 90),
        "% Below start": np.mean(b < current_balance) * 100,
        "% At $0 (ruin)": np.mean(b <= 0) * 100,
        "% Reached goal": np.mean((~np.isnan(t_goal)) & (t_goal <= Y)) * 100,
        "% Working by now": np.mean((~np.isnan(t_work_arr)) & (t_work_arr <= Y)) * 100,
    })
table = pd.DataFrame(rows).set_index("Year")
money = lambda x: f"${x:,.0f}"
pctf = lambda x: f"{x:.1f}%"
display(table.style
    .format({"P10": money, "P25": money, "Median value": money,
             "P75": money, "P90": money,
             "% Below start": pctf, "% At $0 (ruin)": pctf,
             "% Reached goal": pctf, "% Working by now": pctf})
    .background_gradient(subset=["% At $0 (ruin)"], cmap="Reds", vmin=0, vmax=50)
    .background_gradient(subset=["% Reached goal"], cmap="Greens", vmin=0, vmax=100)
    .background_gradient(subset=["% Working by now"], cmap="Oranges", vmin=0, vmax=100)
    .background_gradient(subset=["Median value"], cmap="Blues")
    .set_caption(f"Start ${current_balance:,.0f} | {leverage_mode} (avg {res['avg_lev']:.2f}x) | "
                 f"band +{band_pct_up*100:.0f}%/-{band_pct_down*100:.0f}% | "
                 f"margin {'ON' if use_margin else 'OFF'} | {n_paths} paths")
    .set_table_styles([{"selector": "caption",
                        "props": [("font-size", "14px"), ("font-weight", "bold"),
                                  ("text-align", "left"), ("padding-bottom", "8px")]}]))

# ============================================================
# 6. HEADLINE
# ============================================================
final = bal_yearly[:, -1]
print("\n" + "=" * 70)
print(f"HEADLINE @ year {horizon_years}   [{leverage_mode}]")
print("=" * 70)
print(f"Median ending value:  ${np.median(final):,.0f}")
print(f"Middle 50% (P25-P75): ${np.percentile(final,25):,.0f} - ${np.percentile(final,75):,.0f}")
print(f"Middle 80% (P10-P90): ${np.percentile(final,10):,.0f} - ${np.percentile(final,90):,.0f}")
print(f"Chance of ruin:       {np.mean(final<=0)*100:.1f}%")
print(f"Chance below start:   {np.mean(final<current_balance)*100:.1f}%")
print(f"Reached ${goal_target/1e6:.2f}M goal:  "
      f"{np.mean((~np.isnan(t_goal)) & (t_goal<=horizon_years))*100:.1f}%")
reached = t_goal[~np.isnan(t_goal)]
if len(reached):
    print(f"Years to goal:        median {np.median(reached):.1f} | "
          f"P25 {np.percentile(reached,25):.1f} | P75 {np.percentile(reached,75):.1f} | "
          f"P90 {np.percentile(reached,90):.1f}")
print(f"Average leverage while invested: {res['avg_lev']:.2f}x")

# ============================================================
# 6a. STATIC vs DYNAMIC (same random paths)
# ============================================================
def summary(rr, label):
    f = rr["bal_yearly"][:, -1]; tg = rr["t_goal"]
    hit = tg[~np.isnan(tg)]
    return pd.Series({
        "Median ending value": np.median(f),
        "P10 ending value": np.percentile(f, 10),
        "P25 ending value": np.percentile(f, 25),
        "P75 ending value": np.percentile(f, 75),
        "P90 ending value": np.percentile(f, 90),
        "Chance of ruin %": np.mean(f <= 0) * 100,
        "Chance below start %": np.mean(f < current_balance) * 100,
        f"Reached ${goal_target/1e6:.2f}M goal %": np.mean(~np.isnan(tg)) * 100,
        "Median years to goal": np.median(hit) if len(hit) else np.nan,
        "Median worst drawdown %": np.median(rr["maxdd"]) * 100,
        "Paths with margin call %": np.mean(rr["margin_call"]) * 100,
        "Median interest paid": np.median(rr["interest"]),
        "Avg leverage while invested": rr["avg_lev"],
    }, name=label)


if res_static is not None:
    cmp_df = pd.concat([summary(res_static, f"STATIC {leverage}x"), summary(res, leverage_mode)], axis=1)
    cmp_df["Difference"] = cmp_df.iloc[:, 1] - cmp_df.iloc[:, 0]
    print("\n" + "=" * 70); print("STATIC vs DYNAMIC  (identical random paths)"); print("=" * 70)
    fmt = cmp_df.copy().astype(object)
    for idx in fmt.index:
        for c in fmt.columns:
            val = cmp_df.loc[idx, c]
            if "value" in idx or "interest" in idx:
                fmt.loc[idx, c] = f"${val:,.0f}"
            elif "leverage" in idx.lower():
                fmt.loc[idx, c] = f"{val:.2f}x"
            elif "years" in idx:
                fmt.loc[idx, c] = f"{val:.1f}"
            else:
                fmt.loc[idx, c] = f"{val:.1f}"
    display(fmt)

# ============================================================
# 6b. WORK SUMMARY
# ============================================================
print("\n" + "=" * 70); print("WORK REQUIREMENT"); print("=" * 70)
if monthly_income <= 0 or work_starts == "never":
    print("No job modeled.")
else:
    needed = work_years_arr > 0
    print(f"Paths where you work at all:  {needed.mean()*100:.1f}%")
    if needed.any():
        wy = work_years_arr[needed]
        print(f"Years worked (those paths):   median {np.median(wy):.1f} | "
              f"P75 {np.percentile(wy,75):.1f} | P90 {np.percentile(wy,90):.1f} | max {wy.max():.1f}")
        print(f"Separate work stints:         median {np.median(work_spells_arr[needed]):.0f} | "
              f"max {work_spells_arr[needed].max():.0f}")
        tw = t_work_arr[~np.isnan(t_work_arr)]
        if len(tw):
            print(f"Work first starts in year:    median {np.median(tw):.1f} | "
                  f"P10 {np.percentile(tw,10):.1f} | P90 {np.percentile(tw,90):.1f}")
        gi = work_income_arr[needed]
        print(f"Gross income earned:          median ${np.median(gi):,.0f} | "
              f"P90 ${np.percentile(gi,90):,.0f}")
        ca = contrib_arr[needed]
        print(f"Surplus invested (contrib.):  median ${np.median(ca):,.0f} | "
              f"P10 ${np.percentile(ca,10):,.0f} | P90 ${np.percentile(ca,90):,.0f}")
    print(f"Average years worked (all paths): {work_years_arr.mean():.2f}")

# ============================================================
# 6c. BORROWING SUMMARY
# ============================================================
print("\n" + "=" * 70); print("BORROWING (MARGIN / SECURE OUTSIDE DEBT)"); print("=" * 70)
if not use_margin:
    print("Margin disabled — every withdrawal sold shares.")
else:
    used = peak_margin_arr > 0
    print(f"Paths that used margin:       {used.mean()*100:.1f}%")
    if used.any():
        pm = peak_margin_arr[used]
        print(f"Peak margin balance:          median ${np.median(pm):,.0f} | "
              f"P90 ${np.percentile(pm,90):,.0f} | max ${pm.max():,.0f}")
    if use_outside_debt:
        uo = peak_outside_arr > 0
        print(f"Paths that used outside debt: {uo.mean()*100:.1f}%")
        if uo.any():
            po = peak_outside_arr[uo]
            print(f"Peak outside debt balance:    median ${np.median(po):,.0f} | "
                  f"P90 ${np.percentile(po,90):,.0f} | "
                  f"hit the ${outside_debt_limit:,.0f} cap in "
                  f"{(peak_outside_arr>=outside_debt_limit*0.99).mean()*100:.1f}% of paths")
    print(f"Paths with a MARGIN CALL:     {margin_call_arr.mean()*100:.1f}%")
    print(f"Lifetime interest paid:       median ${np.median(interest_arr):,.0f} | "
          f"P90 ${np.percentile(interest_arr,90):,.0f}")

# ============================================================
# 7. CHARTS
# ============================================================
plt.rcParams["figure.facecolor"] = plt.rcParams["axes.facecolor"] = "white"
fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
x_years = np.arange(nd + 1) / 252
yr_pts = np.concatenate([[0], years])


def band(q, by=None):
    by = bal_yearly if by is None else by
    return np.concatenate([[current_balance], np.percentile(by, q, axis=0)])


money_fmt = mticker.FuncFormatter(
    lambda v, _: f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}K" if v >= 1e3 else f"${v:.0f}")


def draw_equity(ax, logscale):
    for p in range(n_plot_paths):
        ax.plot(x_years, bal_daily_plot[p], color="#4C72B0", alpha=0.10, linewidth=0.8)
    ax.fill_between(yr_pts, band(10), band(90), color="#C44E52", alpha=0.12, label="10th-90th (80% of paths)")
    ax.fill_between(yr_pts, band(25), band(75), color="#C44E52", alpha=0.30, label="25th-75th (middle 50%)")
    ax.plot(yr_pts, band(50), color="#C44E52", linewidth=2.5, label="Median")
    if res_static is not None:
        ax.plot(yr_pts, band(50, res_static["bal_yearly"]), color="#555555", linewidth=1.8, ls="--",
                label=f"Median, static {leverage}x")
    ax.axhline(current_balance, color="gray", ls="--", lw=1, alpha=0.6, label="Start")
    if work_starts == "if_portfolio_falls_to" and monthly_income > 0:
        ax.axhline(work_trigger_balance, color="#DD8452", ls=":", lw=1.6,
                   label=f"Work trigger ${work_trigger_balance/1e3:.0f}K")
    ax.axhline(goal_target, color="#55A868", ls=":", lw=1.6, label=f"Goal ${goal_target/1e6:.2f}M")
    ax.yaxis.set_major_formatter(money_fmt)
    ax.set_xlabel("Years")
    ax.set_xlim(0, horizon_years)
    if logscale:
        ax.set_yscale("log")
        ax.set_ylabel("Equity, net of debt (log)")
        ax.set_title(f"Equity curves - LOG scale ({leverage_mode})")
        ax.grid(True, which="both", alpha=0.2)
    else:
        top = max(band(linear_chart_cap_pct).max(), goal_target, current_balance) * linear_chart_headroom
        ax.set_ylim(0, top)
        n_off = int(np.sum(np.nanmax(bal_daily_plot, axis=1) > top))
        ax.set_ylabel("Equity, net of debt")
        ax.set_title("Equity curves - LINEAR scale" +
                     (f" ({n_off} of {n_plot_paths} paths run off the top)" if n_off else ""))
        ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)


draw_equity(axes[0, 0], logscale=True)
draw_equity(axes[0, 1], logscale=False)

ax2 = axes[1, 0]
fc = np.clip(final, 1, None)
ax2.hist(fc, bins=np.logspace(np.log10(max(fc.min(), 1)), np.log10(fc.max() + 1), 40),
         color="#4C72B0", alpha=0.75, edgecolor="white")
ax2.set_xscale("log")
ax2.axvspan(np.percentile(final, 25), np.percentile(final, 75), color="#C44E52", alpha=0.12, label="Middle 50%")
ax2.axvline(current_balance, color="gray", ls="--", lw=1.2, label="Start")
ax2.axvline(np.median(final), color="#C44E52", lw=1.5, label="Median")
if res_static is not None:
    ax2.axvline(np.median(res_static["bal_yearly"][:, -1]), color="#555555", lw=1.5, ls="--",
                label=f"Median, static {leverage}x")
ax2.xaxis.set_major_formatter(money_fmt)
ax2.set_xlabel("Ending equity (log)")
ax2.set_ylabel("Paths")
ax2.set_title(f"Ending balance, year {horizon_years}")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)

ax3 = axes[1, 1]
if res_static is not None:
    qs = np.arange(5, 96, 5)
    dyn_q = np.percentile(final, qs)
    sta_q = np.percentile(res_static["bal_yearly"][:, -1], qs)
    ax3.plot(qs, dyn_q, color="#C44E52", lw=2.2, marker="o", ms=4, label=leverage_mode)
    ax3.plot(qs, sta_q, color="#555555", lw=1.8, ls="--", marker="o", ms=4, label=f"static {leverage}x")
    ax3.axhline(current_balance, color="gray", ls=":", lw=1)
    ax3.yaxis.set_major_formatter(money_fmt)
    ax3.set_xlabel("Percentile of outcomes (5 = bad luck, 95 = good luck)")
    ax3.set_ylabel(f"Ending equity, year {horizon_years}")
    ax3.set_title("Same random paths: dynamic vs static, every percentile")
    ax3.legend(fontsize=9, loc="upper left")
elif use_margin and (peak_margin_arr > 0).any():
    pm = peak_margin_arr[peak_margin_arr > 0]
    ax3.hist(pm, bins=30, color="#8172B2", alpha=0.8, edgecolor="white")
    ax3.axvline(np.median(pm), color="#C44E52", lw=1.5, label=f"Median ${np.median(pm)/1e3:.0f}K")
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))
    ax3.set_xlabel("Peak margin balance"); ax3.set_ylabel("Paths")
    ax3.set_title(f"Margin used ({(peak_margin_arr==0).mean()*100:.0f}% never borrowed)")
    ax3.legend(fontsize=9)
else:
    ax3.text(0.5, 0.5, "No margin used", ha="center", va="center", fontsize=13, color="gray")
    ax3.set_xticks([]); ax3.set_yticks([])
ax3.grid(True, alpha=0.2)

plt.tight_layout()
if os.environ.get("QQQ_SAVEFIG"):
    fig.savefig(os.environ["QQQ_SAVEFIG"], dpi=110, bbox_inches="tight")
plt.show()

print("\nAssumptions: post-2005 bootstrap; QLD cost ~5.45%/yr on the leveraged sleeve; "
      f"SGOV {sgov_annual_yield*100:.1f}%; margin {margin_rate_annual*100:.1f}%; "
      f"secure outside debt {outside_debt_rate_annual*100:.1f}%; trading cost {trade_cost_bps} bp "
      "per unit of exposure changed; withdrawals nominal.")
print("Balances shown are equity NET of margin and secure outside debt.")
print("This is a statistical model of one strategy -- not financial advice.")
