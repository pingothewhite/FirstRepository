# %% [markdown]
# # QQQ Bayesian Forward-Return Model (3 / 6 / 9 / 12 months)
#
# **Question:** given where QQQ sits on the chart *today*, what has history said about
# its return over the next 3, 6, 9 and 12 months?
#
# **Three Bayesian layers**, all built only from information known on each day (no look-ahead):
#
# 1. **Single-signal Bayes tables** – Bayes' rule written out explicitly:
#    `P(Up | signal bin) = P(signal bin | Up) · P(Up) / P(signal bin)`
# 2. **Naive Bayes classifier** – combines all signals as a sum of log-likelihood ratios:
#    `log O(Up | x) = log O(Up) + Σ_i log[ P(x_i | Up) / P(x_i | Down) ]`
# 3. **Bayesian analog model (Normal–Inverse-Gamma)** – finds historical days whose chart
#    state resembles today, weights them by similarity, and updates a conjugate prior on
#    the mean & variance of the forward log return. The posterior predictive is a
#    Student-t, which gives P(up), expected return, and credible intervals.
#    Overlapping forward windows are *not* independent, so the sample size is shrunk to
#    an effective number of independent horizon-length blocks.
#
# Every model is then **walk-forward tested** (trained only on data whose outcome was
# already known at the time) and scored against a no-skill "base rate" forecast with the
# Brier skill score. That test is what tells you whether the model is actually predictive.
#
# **Colab usage:** upload `QQQ-history.xlsx` (or leave `EXTEND_WITH_YFINANCE=True`
# to pull QQQ back to its 1999 launch) and Runtime → Run all.

# %%
# ---------------------------------------------------------------- configuration
import os, glob, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

DATA_XLSX = os.environ.get("QQQ_XLSX", "")      # path to your export; auto-detected if blank
EXTEND_WITH_YFINANCE = True                      # prepend QQQ history back to 1999 (needs internet)
YF_START = "1999-03-10"                          # QQQ inception
HORIZONS = {"3m": 63, "6m": 126, "9m": 189, "12m": 252}   # trading days
N_BINS = 5                                       # quantile bins per signal (quintiles)
LAPLACE = 1.0                                    # Dirichlet/Laplace smoothing for likelihoods
ANALOG_FRAC = 0.10                               # share of history used as nearest analogs
PRIOR_KAPPA = 2.0                                # prior worth ~2 independent periods
PRIOR_ALPHA = 3.0
WF_STEP = 21                                     # walk-forward: re-forecast every ~month
WF_MIN_TRAIN_YEARS = 4                           # walk-forward: min history before first test
OUT_DIR = os.environ.get("QQQ_OUT", ".")

FEATURES = {
    "dist_200dma": "Price vs 200-day avg (%)",
    "drawdown":    "Distance from all-time high (%)",
    "mom_12m":     "Trailing 12-month return (%)",
    "rsi_14":      "RSI(14)",
    "vol_21d":     "21-day realized vol (ann. %)",
}
SHORT = dict(zip(FEATURES.values(), ["vs 200-day avg", "from ATH", "12m momentum", "RSI(14)", "21d vol"]))

# %%
# ---------------------------------------------------------------- data loading
def find_xlsx():
    if DATA_XLSX and os.path.exists(DATA_XLSX):
        return DATA_XLSX
    hits = sorted(glob.glob("*QQQ*history*.xlsx") + glob.glob("/content/*QQQ*.xlsx"))
    if hits:
        return hits[0]
    try:                                        # Colab: prompt for upload
        from google.colab import files
        up = files.upload()
        return next(iter(up))
    except Exception:
        return None


def load_xlsx(path):
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    col = "Adj. Close" if "Adj. Close" in df else "Close"   # adjusted = total return
    s = df.set_index("Date")[col].astype(float).sort_index()
    return s[~s.index.duplicated()]


def load_yf():
    try:
        import yfinance as yf
    except ImportError:
        os.system("pip -q install yfinance")
        import yfinance as yf
    d = yf.download("QQQ", start=YF_START, auto_adjust=True, progress=False)
    c = d["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
    return c.dropna().astype(float)


def load_prices():
    path = find_xlsx()
    s = load_xlsx(path) if path else None
    if s is not None:
        print(f"Loaded {path}: {s.index[0].date()} → {s.index[-1].date()} ({len(s)} rows)")
    if EXTEND_WITH_YFINANCE:
        try:
            y = load_yf()
            print(f"yfinance QQQ: {y.index[0].date()} → {y.index[-1].date()} ({len(y)} rows)")
            if s is None:
                s = y
            else:
                # splice: keep your file as the authority, scale older yfinance data to join
                join = s.index[0]
                older = y[y.index < join]
                ref = y[y.index >= join]
                if len(older) and len(ref):
                    scale = s.iloc[0] / ref.iloc[0]
                    s = pd.concat([older * scale, s]).sort_index()
                    newer = y[y.index > s.index[-1]]   # anything after your export
                    if len(newer):
                        s = pd.concat([s, newer * (s.iloc[-1] / y.loc[:s.index[-1]].iloc[-1])])
                print(f"Spliced series: {s.index[0].date()} → {s.index[-1].date()} ({len(s)} rows)")
        except Exception as e:
            print(f"yfinance unavailable ({e.__class__.__name__}); using file only.")
    if s is None:
        raise SystemExit("No data: upload the QQQ xlsx or enable yfinance.")
    return s


px = load_prices()

# %%
# ---------------------------------------------------------------- chart-state features
def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def build_frame(px):
    f = pd.DataFrame({"px": px})
    lr = np.log(px).diff()
    f["dist_200dma"] = 100 * (px / px.rolling(200).mean() - 1)
    f["drawdown"] = 100 * (px / px.cummax() - 1)
    f["mom_12m"] = 100 * (px / px.shift(252) - 1)
    f["rsi_14"] = rsi(px)
    f["vol_21d"] = 100 * lr.rolling(21).std() * np.sqrt(252)
    for k, h in HORIZONS.items():
        f[f"fwd_{k}"] = np.log(px.shift(-h) / px)        # forward log return (target)
    return f.dropna(subset=list(FEATURES))


F = build_frame(px)
today = F.iloc[-1]
T0 = F.index[-1]
print(f"\nAs-of {T0.date()}  QQQ = {today.px:,.2f}")
for k, lab in FEATURES.items():
    pct = (F[k] <= today[k]).mean() * 100
    print(f"  {lab:34s} {today[k]:8.2f}   (historical percentile {pct:5.1f})")
print("\nFeature correlation:\n", F[list(FEATURES)].corr().round(2))

# %%
# ---------------------------------------------------------------- Bayesian machinery
def bin_edges(x, n=N_BINS):
    e = np.unique(np.quantile(x, np.linspace(0, 1, n + 1)))
    e[0], e[-1] = -np.inf, np.inf
    return e


def naive_bayes(train, x_now, target, feats=FEATURES):
    """Returns prior, posterior P(up), and per-feature log-likelihood ratios."""
    y = (train[target] > 0).values
    n_up, n_dn = y.sum(), (~y).sum()
    prior = (n_up + LAPLACE) / (len(y) + 2 * LAPLACE)
    logodds = np.log(prior / (1 - prior))
    rows = []
    for k in feats:
        e = bin_edges(train[k].values)
        b = np.digitize(train[k].values, e[1:-1])
        b_now = int(np.digitize([x_now[k]], e[1:-1])[0])
        K = len(e) - 1
        p_up = (np.sum(b[y] == b_now) + LAPLACE) / (n_up + K * LAPLACE)   # P(bin | Up)
        p_dn = (np.sum(b[~y] == b_now) + LAPLACE) / (n_dn + K * LAPLACE)  # P(bin | Down)
        p_bin = p_up * prior + p_dn * (1 - prior)                        # P(bin)
        post1 = p_up * prior / p_bin                                     # Bayes' rule
        llr = np.log(p_up / p_dn)
        logodds += llr
        rows.append(dict(feature=feats[k], bin=f"{b_now + 1}/{K}",
                         lo=e[b_now], hi=e[b_now + 1],
                         **{"P(bin|Up)": p_up, "P(bin|Down)": p_dn, "P(bin)": p_bin,
                            "P(Up|bin)": post1, "LLR": llr}))
    post = 1 / (1 + np.exp(-logodds))
    return prior, post, pd.DataFrame(rows)


def block_neff(dates, w, h):
    """Effective # of independent observations: group weights into horizon-length
    blocks of trading days (overlapping windows inside a block ≈ one draw)."""
    idx = np.searchsorted(ALL_DATES, dates.values)
    blk = idx // h
    B = pd.Series(w).groupby(blk).sum().values
    return B.sum() ** 2 / np.sum(B ** 2)


def analog_nig(train, x_now, target, h, feats=list(FEATURES)):
    """Kernel-weighted analogs + Normal–Inverse-Gamma conjugate update.
    Returns posterior-predictive Student-t for the forward log return."""
    X = train[feats].values
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    z0 = (np.array([x_now[k] for k in feats]) - mu) / sd
    d = np.sqrt(((Z - z0) ** 2).sum(1))
    k = max(int(len(d) * ANALOG_FRAC), 30)
    r = np.sort(d)[k - 1]
    w = np.where(d <= r, (1 - (d / r) ** 3) ** 3, 0.0)                  # tricube kernel
    y = train[target].values
    m = w > 0
    w, yv, dates = w[m], y[m], train.index[m]
    xbar = np.sum(w * yv) / w.sum()
    s2 = np.sum(w * (yv - xbar) ** 2) / w.sum()
    n = block_neff(dates, w, h)

    # prior = unconditional history (same horizon)
    mu0, var0 = y.mean(), y.var()
    k0, a0 = PRIOR_KAPPA, PRIOR_ALPHA
    b0 = var0 * (a0 - 1)                                                 # E[σ²] = var0
    kn = k0 + n
    mun = (k0 * mu0 + n * xbar) / kn
    an = a0 + n / 2
    bn = b0 + 0.5 * n * s2 + k0 * n * (xbar - mu0) ** 2 / (2 * kn)
    df = 2 * an
    scale = np.sqrt(bn * (kn + 1) / (an * kn))
    t = stats.t(df, loc=mun, scale=scale)
    return dict(dist=t, mu_post=mun, n_eff=n, n_analogs=int(m.sum()),
                analog_mean=xbar, analog_sd=np.sqrt(s2), prior_mu=mu0,
                analog_dates=dates, analog_w=w, analog_y=yv,
                p_up=1 - t.cdf(0))


ALL_DATES = F.index.values

# %%
# ---------------------------------------------------------------- 1) explicit Bayes tables & 2) naive Bayes
nb_now, nig_now = {}, {}
for k, h in HORIZONS.items():
    tgt = f"fwd_{k}"
    train = F.dropna(subset=[tgt])
    prior, post, tab = naive_bayes(train, today, tgt)
    nb_now[k] = (prior, post, tab)
    print(f"\n=== {k} horizon ({h} trading days) — {len(train)} labelled days, "
          f"~{len(train) // h} independent windows ===")
    print(f"Prior P(Up) = {prior:.3f}")
    print(tab.drop(columns=["lo", "hi"]).round(3).to_string(index=False))
    print(f"Naive-Bayes posterior P(Up | all signals) = {post:.3f}   "
          f"(Σ LLR = {tab.LLR.sum():+.3f})")

# %%
# ---------------------------------------------------------------- 3) Bayesian analog / NIG posterior predictive
rows = []
for k, h in HORIZONS.items():
    tgt = f"fwd_{k}"
    train = F.dropna(subset=[tgt])
    res = analog_nig(train, today, tgt, h)
    nig_now[k] = res
    t = res["dist"]
    q = t.ppf([0.10, 0.25, 0.5, 0.75, 0.90])
    rows.append({
        "horizon": k,
        "analogs": res["n_analogs"],
        "n_eff (indep.)": round(res["n_eff"], 1),
        "prior mean %": 100 * (np.exp(res["prior_mu"]) - 1),
        "analog mean %": 100 * (np.exp(res["analog_mean"]) - 1),
        "posterior median %": 100 * (np.exp(q[2]) - 1),
        "P(up)": res["p_up"],
        "P(< -10%)": t.cdf(np.log(0.9)),
        "P(> +10%)": 1 - t.cdf(np.log(1.1)),
        "10th pct $": today.px * np.exp(q[0]),
        "median $": today.px * np.exp(q[2]),
        "90th pct $": today.px * np.exp(q[4]),
    })
summary = pd.DataFrame(rows).set_index("horizon")
print("\nBayesian analog model — posterior predictive for QQQ from", T0.date())
print(summary.round(3).to_string())

# %%
# ---------------------------------------------------------------- walk-forward (out-of-sample) test
def walk_forward():
    out = []
    start = F.index[0] + pd.DateOffset(years=WF_MIN_TRAIN_YEARS)
    test_idx = np.arange(0, len(F), WF_STEP)
    for k, h in HORIZONS.items():
        tgt = f"fwd_{k}"
        for i in test_idx:
            t0 = F.index[i]
            if t0 < start or np.isnan(F[tgt].iloc[i]):
                continue
            # only days whose outcome was fully known by t0
            train = F.iloc[: max(i - h, 0)].dropna(subset=[tgt])
            if len(train) < 3 * h:
                continue
            x = F.iloc[i]
            base = (train[tgt] > 0).mean()
            _, p_nb, _ = naive_bayes(train, x, tgt)
            r = analog_nig(train, x, tgt, h)
            out.append(dict(h=k, date=t0, y=float(F[tgt].iloc[i] > 0), ret=F[tgt].iloc[i],
                            p_base=base, p_nb=p_nb, p_nig=r["p_up"], mu_nig=r["mu_post"],
                            mu_base=train[tgt].mean()))
    return pd.DataFrame(out)


def brier(p, y):
    return np.mean((p - y) ** 2)


wf = walk_forward()
score_rows = []
for k, h in HORIZONS.items():
    g = wf[wf.h == k]
    if g.empty:
        continue
    bb = brier(g.p_base, g.y)
    score_rows.append({
        "horizon": k,
        "tests": len(g),
        "~indep.": round(len(g) * WF_STEP / h, 1),
        "actual up-rate": g.y.mean(),
        "Brier base": bb,
        "Brier NB": brier(g.p_nb, g.y),
        "Brier analog": brier(g.p_nig, g.y),
        "BSS NB": 1 - brier(g.p_nb, g.y) / bb,
        "BSS analog": 1 - brier(g.p_nig, g.y) / bb,
        "IC analog (rank corr)": stats.spearmanr(g.mu_nig - g.mu_base, g.ret)[0],
        "hit-rate NB": np.mean((g.p_nb > 0.5) == (g.y == 1)),
    })
scores = pd.DataFrame(score_rows).set_index("horizon")
print("\nWalk-forward out-of-sample scores "
      "(BSS > 0 means the model beat the base rate; ~indep. = effective # of test windows)")
print(scores.round(3).to_string())

# %%
# ---------------------------------------------------------------- what past analogs actually did
k12 = nig_now["12m"]
ad = pd.DataFrame({"date": k12["analog_dates"], "w": k12["analog_w"],
                   "ret12m_%": 100 * (np.exp(k12["analog_y"]) - 1)})
ad["year"] = ad.date.dt.year
by_year = ad.groupby("year").apply(lambda g: pd.Series({
    "analog days": len(g), "weight share %": 100 * g.w.sum() / ad.w.sum(),
    "avg fwd 12m %": np.average(g["ret12m_%"], weights=g.w)}))
print("\nWhere today's closest 12-month analogs come from:")
print(by_year.round(1).to_string())

# %%
# ---------------------------------------------------------------- charts
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titleweight": "bold", "axes.titlecolor": INK})

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.32, wspace=0.28)

# (a) price + fan chart of posterior predictive quantiles
ax = fig.add_subplot(gs[0, :])
hist = px[px.index >= T0 - pd.DateOffset(years=3)]
ax.plot(hist.index, hist.values, color=INK, lw=1.6, label="QQQ (adj.)")
ax.plot(hist.index, px.rolling(200).mean().reindex(hist.index), color=INK2, lw=1.2,
        ls="--", label="200-day avg")
fut_t = [T0] + [T0 + pd.tseries.offsets.BDay(h) for h in HORIZONS.values()]
def qline(q): return [today.px] + [today.px * np.exp(nig_now[k]["dist"].ppf(q)) for k in HORIZONS]
ax.fill_between(fut_t, qline(0.10), qline(0.90), color=C_BLUE, alpha=0.15, lw=0,
                label="80% credible band")
ax.fill_between(fut_t, qline(0.25), qline(0.75), color=C_BLUE, alpha=0.28, lw=0,
                label="50% credible band")
ax.plot(fut_t, qline(0.5), color=C_BLUE, lw=2, marker="o", ms=5, label="posterior median")
for k, t_ in zip(HORIZONS, fut_t[1:]):
    ax.annotate(f"{k}\n${today.px * np.exp(nig_now[k]['dist'].ppf(0.5)):,.0f}",
                (t_, today.px * np.exp(nig_now[k]["dist"].ppf(0.5))), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9, color=INK)
ax.set_title(f"QQQ {today.px:,.2f} on {T0.date()} — Bayesian analog forecast")
ax.set_ylabel("Price ($)")
ax.legend(loc="upper left", frameon=False, fontsize=9)

# (b) P(up): prior vs naive Bayes vs analog
ax = fig.add_subplot(gs[1, 0])
xs = np.arange(len(HORIZONS))
wdt = 0.26
vals = [[nb_now[k][0] for k in HORIZONS], [nb_now[k][1] for k in HORIZONS],
        [nig_now[k]["p_up"] for k in HORIZONS]]
for j, (v, c, lab) in enumerate(zip(vals, [INK2, C_ORANGE, C_BLUE],
                                    ["Base rate (prior)", "Naive Bayes", "Analog NIG"])):
    b = ax.bar(xs + (j - 1) * wdt, v, wdt - 0.03, color=c, label=lab)
    for rect, val in zip(b, v):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.01, f"{val:.0%}",
                ha="center", fontsize=8, color=INK)
ax.axhline(0.5, color=INK2, lw=0.8)
ax.set_xticks(xs, list(HORIZONS))
ax.set_ylim(0, 1.22)
ax.set_yticks(np.linspace(0, 1, 6))
ax.set_title("P(QQQ higher at horizon)")
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)

# (c) naive-Bayes evidence (log-likelihood ratios) for 12m
ax = fig.add_subplot(gs[1, 1])
tab = nb_now["12m"][2]
cols = [C_AQUA if v > 0 else C_ORANGE for v in tab.LLR]
ax.barh(tab.feature.map(SHORT), tab.LLR, color=cols, height=0.6)
ax.axvline(0, color=INK2, lw=0.8)
for yv, (v, bn) in enumerate(zip(tab.LLR, tab["bin"])):
    ax.text(v + (0.01 if v >= 0 else -0.01), yv, f"{v:+.2f} (bin {bn})",
            va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK)
ax.set_title("12m evidence: log[P(signal|Up)/P(signal|Down)]")
ax.invert_yaxis()
lim = max(0.3, np.abs(tab.LLR).max() * 1.8)
ax.set_xlim(-lim, lim)

os.makedirs(OUT_DIR, exist_ok=True)
png = os.path.join(OUT_DIR, "qqq_bayes_forecast.png")
fig.savefig(png, dpi=130, bbox_inches="tight")
print(f"\nSaved chart → {png}")
plt.show()

# %%
# ---------------------------------------------------------------- plain-English verdict
print("\n" + "=" * 78)
print(f"VERDICT as of {T0.date()} (QQQ {today.px:,.2f}; data {F.index[0].date()}→{T0.date()})")
print("=" * 78)
for k in HORIZONS:
    r = nig_now[k]
    t = r["dist"]
    skill = scores.loc[k, "BSS analog"] if k in scores.index else np.nan
    print(f"{k:>4}: P(up)={r['p_up']:.0%} (base rate {nb_now[k][0]:.0%}), "
          f"median {100 * (np.exp(t.ppf(0.5)) - 1):+.1f}%, "
          f"80% band {100 * (np.exp(t.ppf(0.1)) - 1):+.0f}% … {100 * (np.exp(t.ppf(0.9)) - 1):+.0f}%, "
          f"out-of-sample skill BSS={skill:+.2f}")
print("\nRead BSS as: >0 beat the base rate, ≤0 no better than just using the base rate.")
