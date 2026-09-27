# Does market breadth predict QQQ?  One-cell Colab backtest of the "Market Breadth Monitor" indicators.
# Paste this whole thing into ONE Colab cell and press play (Shift+Enter). First run takes ~5-10 minutes
# (downloads ~12 years of daily bars for ~1,500 stocks); re-runs the same day use the cache.

!pip -q install -U yfinance plotly statsmodels

START_DATE = "2014-06-01"      # data download start (a year+ of warm-up for 52-week highs, 50dma, etc.)
ANALYSIS_START = "2016-09-27"  # the 10-year test window starts here
HORIZONS = [5, 10, 21, 63]     # forward QQQ returns tested, in trading days (1 week, 2 weeks, 1 month, 1 quarter)
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 2e6
REQUIRE_VOLUME_CONFIRM = True
CACHE_FILE = "breadth_backtest_prices.pkl"
FORCE_REDOWNLOAD = False

WEIGHTS = {
    "5-day ratio": 0.10, "10-day ratio": 0.10, "Quarter (±25%)": 0.15, "Month (±25%)": 0.10,
    "34-day (±13%)": 0.10, "% above 50dma": 0.15, "New highs vs lows": 0.10, "Advance/decline": 0.10,
    "50dma trend (5d)": 0.05, "S&P trend": 0.05,
}

import io, os, time, json, pickle, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from IPython.display import HTML, display
warnings.filterwarnings("ignore")
RNG = np.random.default_rng(7)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

# ============================================================== 1. data
def _wiki_symbols(page):
    html = requests.get(f"https://en.wikipedia.org/wiki/{page}", headers=UA, timeout=30).text
    for t in pd.read_html(io.StringIO(html)):
        for col in t.columns:
            name = str(col).lower()
            if "symbol" in name or "ticker" in name:
                s = t[col].astype(str).str.strip()
                s = s[s.str.fullmatch(r"[A-Z][A-Z.\-]{0,6}")]
                if len(s) > 50:
                    return s.tolist()
    raise RuntimeError(f"no ticker table found on {page}")

def sp1500():
    out = []
    for p in ["List_of_S%26P_500_companies", "List_of_S%26P_400_companies", "List_of_S%26P_600_companies"]:
        out += _wiki_symbols(p)
    return sorted({s.replace(".", "-").upper() for s in out})

def _clean_index(df):
    df = df.copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()

def _one(ticker):
    d = _clean_index(yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False))
    return d["Close"].iloc[:, 0] if isinstance(d["Close"], pd.DataFrame) else d["Close"]

def download_prices(tickers, chunk=100, pause=1.0):
    frames = []
    for i in range(0, len(tickers), chunk):
        batch, d = tickers[i:i + chunk], None
        for attempt in range(4):
            try:
                d = yf.download(batch, start=START_DATE, interval="1d", auto_adjust=True,
                                group_by="column", threads=True, progress=False)
                break
            except Exception as e:
                print(f"\n  batch {i // chunk + 1}: retry {attempt + 1} ({e})")
                time.sleep(5 * (attempt + 1))
        if d is not None and len(d):
            if not isinstance(d.columns, pd.MultiIndex):
                d.columns = pd.MultiIndex.from_product([d.columns, batch])
            frames.append(_clean_index(d))
        print(f"  downloaded {min(i + chunk, len(tickers)):,}/{len(tickers):,}", end="\r")
        time.sleep(pause)
    raw = pd.concat(frames, axis=1)
    return raw.loc[:, ~raw.columns.duplicated()]

def load_data():
    key = (START_DATE, datetime.now().strftime("%Y-%m-%d"))
    if os.path.exists(CACHE_FILE) and not FORCE_REDOWNLOAD:
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        if cached.get("key") == key:
            print("Using prices cached earlier today.")
            return cached["px"]
    tickers = sp1500()
    print(f"Universe: {len(tickers):,} current S&P 1500 members. Downloading daily bars since {START_DATE}...")
    raw = download_prices(tickers)
    px = {f: raw[f] for f in ["Close", "High", "Low", "Volume"]}
    good = px["Close"].notna().sum() >= 60
    px = {f: v.loc[:, good[good].index] for f, v in px.items()}
    px["SPX"] = _one("^GSPC")
    px["QQQ"] = _one("QQQ")          # dividend-adjusted, so returns include dividends
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"key": key, "px": px}, f)
    print(f"\nGot usable data for {good.sum():,} tickers.")
    return px

# ============================================================== 2. breadth (same formulas as the dashboard)
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
    hi252 = H.rolling(252, min_periods=252).max()
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
    b["nh"] = cnt(H >= hi252)
    b["nl"] = cnt(L <= L.rolling(252, min_periods=252).min())
    b["adv_pct"] = b.adv / (b.adv + b.dec).clip(lower=1)
    b["nh_pct"] = b.nh / (b.nh + b.nl).clip(lower=1)
    b["universe"] = elig.sum(axis=1)
    b["hl_universe"] = (elig & hi252.notna()).sum(axis=1)
    spx = px["SPX"].reindex(C.index).ffill()
    b["spx"] = spx
    b["spx_sma50"], b["spx_sma200"] = spx.rolling(50).mean(), spx.rolling(200).mean()
    b["spx_hi252"] = spx.rolling(252, min_periods=1).max()
    b = b[b.universe >= 0.5 * b.universe.rolling(60, min_periods=1).median()]   # drop half-printed days
    return b

def add_score(b):
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
    w = pd.Series(WEIGHTS, dtype=float); w = w / w.sum()
    b = b.copy()
    b["score"] = (1 + 99 * (comp[w.index] * w).sum(axis=1, min_count=len(w))).round()
    return b, comp

# ============================================================== 3. statistics
FEATURES = {   # column: (plain-English name, +1 if the dashboard reads a HIGH value as bullish, -1 if bearish)
    "score":    ("Breadth score (1-100)", +1),
    "r5":       ("5 Day Ratio", +1),
    "r10":      ("10 Day Ratio", +1),
    "up4_f":    ("Stocks Up 4%+ Today", +1),
    "dn4_f":    ("Stocks Down 4%+ Today", -1),
    "net25q":   ("Up minus Down 25%+ Quarter", +1),
    "net25m":   ("Up minus Down 25%+ Month", +1),
    "up50m_f":  ("Up 50%+ Month", +1),
    "dn50m_f":  ("Down 50%+ Month", -1),
    "net13":    ("Up minus Down 13%+ 34 Days", +1),
    "ext10_f":  ("10x ATR Extended", -1),
    "pct50":    ("% above 50dma", +1),
    "pct50_d5": ("5-day change in % above 50dma", +1),
    "adv5":     ("Adv % (5-day average)", +1),
    "adv_pct":  ("Adv % (single day)", +1),
    "nh_pct5":  ("New-high share (5-day)", +1),
    "net_hl":   ("New Highs minus New Lows", +1),
}

def build_features(b):
    f = pd.DataFrame(index=b.index)
    u, hu = b.universe.clip(lower=1), b.hl_universe.clip(lower=1)
    f["score"] = b.score
    f["r5"], f["r10"] = np.log(b.r5.clip(lower=0.05)), np.log(b.r10.clip(lower=0.05))   # ratios -> log scale
    f["up4_f"], f["dn4_f"] = b.up4 / u, b.dn4 / u          # counts -> share of universe (universe size drifts)
    f["net25q"], f["net25m"] = (b.up25q - b.dn25q) / u, (b.up25m - b.dn25m) / u
    f["up50m_f"], f["dn50m_f"] = b.up50m / u, b.dn50m / u
    f["net13"] = (b.up13 - b.dn13) / u
    f["ext10_f"] = b.ext10 / u
    f["pct50"], f["pct50_d5"] = b.pct50, b.pct50.diff(5)
    f["adv5"], f["adv_pct"] = b.adv_pct.rolling(5).mean(), b.adv_pct
    nh5, nl5 = b.nh.rolling(5).sum(), b.nl.rolling(5).sum()
    f["nh_pct5"] = (nh5 + 1) / (nh5 + nl5 + 2)
    f["net_hl"] = (b.nh - b.nl) / hu
    return f

def nw(y, X, lags):
    return sm.OLS(y, sm.add_constant(X), missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": lags})

def spearman(a, b):
    ok = a.notna() & b.notna()
    return a[ok].rank().corr(b[ok].rank()) if ok.sum() > 30 else np.nan

def z(s):
    return (s - s.mean()) / s.std()

def perf(ret):
    ret = ret.dropna()
    eq = (1 + ret).cumprod()
    return {"cagr": eq.iloc[-1] ** (252 / len(ret)) - 1, "vol": ret.std() * np.sqrt(252),
            "sharpe": ret.mean() / ret.std() * np.sqrt(252), "maxdd": (eq / eq.cummax() - 1).min()}

def decluster(mask, gap=20):
    idx, last = [], -10**9
    for i, v in enumerate(mask.values):
        if v and i - last > gap:
            idx.append(i); last = i
    return mask.index[idx]

def analyze(b, qqq):
    feats = build_features(b)
    df = feats.join(qqq.rename("qqq"), how="inner")
    df = df[df.index >= ANALYSIS_START]
    lp = np.log(df.qqq)
    for h in HORIZONS:
        df[f"f{h}"] = lp.shift(-h) - lp            # log return, for regressions
        df[f"s{h}"] = np.exp(df[f"f{h}"]) - 1      # simple return, for tables
    df["p21"], df["p63"], df["p252"] = lp - lp.shift(21), lp - lp.shift(63), lp - lp.shift(252)
    df["vol21"] = lp.diff().rolling(21).std()
    dret = df.qqq.pct_change()
    bb = b.reindex(df.index)
    R = {"events": [], "states": [], "granger": [], "oos": [], "strategies": [], "quint": {}}
    mid = df.index[len(df) // 2]
    R["meta"] = {"start": df.index[0], "end": df.index[-1], "days": len(df), "mid": mid,
                 "u0": int(bb.universe.iloc[0]), "u1": int(bb.universe.iloc[-1]),
                 "qqq_cagr": (df.qqq.iloc[-1] / df.qqq.iloc[0]) ** (252 / (len(df) - 1)) - 1,
                 "same_day": df.adv_pct.corr(lp.diff()), "next_day": df.adv_pct.corr(lp.diff().shift(-1))}
    R["base"] = {h: {"m": df[f"s{h}"].mean(), "p": (df[f"s{h}"] > 0).mean()} for h in HORIZONS}

    # --- indicator scorecard
    rows = []
    for k, (label, sign) in FEATURES.items():
        x = df[k]
        row = {"key": k, "label": label, "sign": sign}
        for h in HORIZONS:
            y = df[f"f{h}"]
            ok = x.notna() & y.notna() & df.p252.notna()
            m = nw(y[ok], z(x[ok]), h)
            ctrl = pd.concat([z(x), z(df.p21), z(df.p63), z(df.p252), z(df.vol21)], axis=1)[ok]
            ctrl.columns = ["x", "p21", "p63", "p252", "vol"]
            mc = nw(y[ok], ctrl, h)
            row.update({f"ic{h}": spearman(x, y), f"t{h}": m.tvalues.iloc[1], f"tc{h}": mc.tvalues["x"],
                        f"ica{h}": spearman(x[:mid], y[:mid]), f"icb{h}": spearman(x[mid:], y[mid:])})
        rows.append(row)
    sc = pd.DataFrame(rows).set_index("key")
    verdicts = []
    for k, r in sc.iterrows():
        best = max([21, 63], key=lambda h: abs(r[f"tc{h}"]) if pd.notna(r[f"tc{h}"]) else 0)
        t, ic, a, c = r[f"tc{best}"], r[f"ic{best}"], r[f"ica{best}"], r[f"icb{best}"]
        consistent = (pd.notna(a) and pd.notna(c) and np.sign(a) == np.sign(c) == np.sign(ic) == np.sign(t))
        x = df[k].dropna()
        if (x != x.mode().iloc[0]).mean() < 0.15:          # e.g. "Down 50%+ Month" is zero most days
            v = "Too rare to test"
        elif pd.notna(t) and abs(t) >= 2 and consistent:
            v = "Real edge"
        elif pd.notna(t) and (abs(t) >= 1.5 or (abs(t) >= 2 and not consistent)):
            v = "Weak / unreliable"
        else:
            v = "No edge"
        way = "as the dashboard assumes" if np.sign(ic) * r.sign > 0 else "opposite to the dashboard, i.e. contrarian"
        higher = "better" if ic > 0 else "worse"
        verdicts.append({"key": k, "verdict": v, "best_h": best, "way": way, "consistent": consistent,
                         "plain": f"Higher readings were followed by {higher} QQQ returns over the next "
                                  f"{'month' if best == 21 else 'quarter'}, {way}."})
    R["scorecard"] = sc.join(pd.DataFrame(verdicts).set_index("key"))

    # --- quintile buckets
    for k in FEATURES:
        x = df[k].dropna()
        qb = pd.qcut(x.rank(method="first"), 5, labels=False)
        rs = []
        for i in range(5):
            sel = qb[qb == i].index
            rs.append({"q": i + 1, "lo": x[sel].min(), "hi": x[sel].max(),
                       **{f"m{h}": df.loc[sel, f"s{h}"].mean() for h in HORIZONS},
                       **{f"p{h}": (df.loc[sel, f"s{h}"].dropna() > 0).mean() for h in HORIZONS}})
        R["quint"][k] = pd.DataFrame(rs)

    # --- dashboard signals as events (first day of each episode; repeats within 20 days ignored)
    u = bb.universe
    ema_adv = b.adv_pct.ewm(span=10, adjust=False).mean().reindex(df.index)
    events = {
        "Short-term overbought: 10-day ratio >= 2": bb.r10 >= 2,
        "Short-term overbought: 5-day ratio >= 2": bb.r5 >= 2,
        "Short-term oversold: 5-day ratio <= 0.5": bb.r5 <= 0.5,
        "Short-term oversold: 10-day ratio <= 0.5": bb.r10 <= 0.5,
        "Heavy selling: 10%+ of stocks down 4% in a day": bb.dn4 >= 0.10 * u,
        "Breadth thrust: 10%+ of stocks up 4% in a day": bb.up4 >= 0.10 * u,
        "Washed out: 20% or fewer above 50dma": bb.pct50 <= 0.20,
        "Stretched: 80%+ above 50dma": bb.pct50 >= 0.80,
        "Froth: many stocks 10x ATR above 50dma": bb.ext10 >= np.maximum(10, 0.005 * u),
        "Divergence: S&P within 3% of high, < 40% above 50dma": (bb.spx >= 0.97 * bb.spx_hi252) & (bb.pct50 < 0.40),
        "Zweig breadth thrust (10d avg adv% from <40% to >61.5%)": (ema_adv > 0.615) & (ema_adv.rolling(10).min() < 0.40),
        "Score <= 20 (strongly bearish)": bb.score <= 20,
        "Score >= 80 (strongly bullish)": bb.score >= 80,
        "Score crosses up through 50": (bb.score >= 50) & (bb.score.shift(1) < 50),
        "Score crosses down through 50": (bb.score < 50) & (bb.score.shift(1) >= 50),
    }
    for label, mask in events.items():
        mask = mask.fillna(False).astype(bool)
        dates = decluster(mask)
        row = {"event": label, "n": len(dates), "on_now": bool(mask.iloc[-1]),
               "last": mask[mask].index[-1] if mask.any() else None, "dates": list(dates)}
        for h in HORIZONS:
            y = df.loc[dates, f"s{h}"].dropna()
            row[f"m{h}"] = y.mean() if len(y) else np.nan
            row[f"p{h}"] = (y > 0).mean() if len(y) else np.nan
            row[f"n{h}"] = len(y)
            if len(y) >= 3:   # how often random dates produce an average this far from normal
                pool = df[f"s{h}"].dropna().values
                sims = RNG.choice(pool, size=(20000, len(y))).mean(axis=1)
                row[f"pv{h}"] = (abs(sims - pool.mean()) >= abs(y.mean() - pool.mean())).mean()
            else:
                row[f"pv{h}"] = np.nan
        R["events"].append(row)
    R["events"] = pd.DataFrame(R["events"])

    # --- regimes: returns while a condition holds vs. doesn't
    states = {
        "Primary trend weak (more stocks down 25% on the quarter than up)": bb.dn25q > bb.up25q,
        "Intermediate trend weak (down-13% > 2x up-13%)": bb.dn13 > 2 * bb.up13,
        "Score >= 60 (bullish or better)": bb.score >= 60,
        "Score < 40 (bearish or worse)": bb.score < 40,
        "Fewer than 40% of stocks above 50dma": bb.pct50 < 0.40,
        "% above 50dma rising over the last 5 days": bb.pct50.diff(5) > 0,
        "S&P above its 200-day average": bb.spx > bb.spx_sma200,
    }
    fvol = lp.diff().rolling(21).std().shift(-21) * np.sqrt(252)
    for label, st in states.items():
        st = st.fillna(False).astype(bool)
        row = {"state": label, "share": st.mean(), "on_now": bool(st.iloc[-1]),
               "vol_on": fvol[st].mean(), "vol_off": fvol[~st].mean()}
        for h in [5, 21, 63]:
            y = df[f"f{h}"]; ok = y.notna()
            row[f"on{h}"], row[f"off{h}"] = df[f"s{h}"][ok & st].mean(), df[f"s{h}"][ok & ~st].mean()
            row[f"pon{h}"], row[f"poff{h}"] = (df[f"s{h}"][ok & st] > 0).mean(), (df[f"s{h}"][ok & ~st] > 0).mean()
            row[f"t{h}"] = nw(y[ok], st[ok].astype(float), h).tvalues.iloc[1]
        R["states"].append(row)
    R["states"] = pd.DataFrame(R["states"])

    # --- Granger causality (weekly): does breadth lead QQQ, or does QQQ lead breadth?
    wk = pd.DataFrame({
        "QQQ weekly return": lp.resample("W-FRI").last().diff(),
        "Breadth score (change)": df.score.resample("W-FRI").last().diff(),
        "% above 50dma (change)": df.pct50.resample("W-FRI").last().diff(),
        "Net 25% quarter (change)": df.net25q.resample("W-FRI").last().diff(),
        "Net 13% / 34 days (change)": df.net13.resample("W-FRI").last().diff(),
        "Adv % (weekly average)": df.adv_pct.resample("W-FRI").mean() - 0.5,
        "New highs minus lows (weekly avg)": df.net_hl.resample("W-FRI").mean(),
        "10 Day Ratio (level)": df.r10.resample("W-FRI").last(),
    }).dropna()
    for var in wk.columns[1:]:
        row = {"var": var}
        for d, cols in [("b2q", ["QQQ weekly return", var]), ("q2b", [var, "QQQ weekly return"])]:
            g = grangercausalitytests(wk[cols], maxlag=4)
            row[d] = min(g[l][0]["ssr_ftest"][1] for l in (1, 2, 4))
            row[d + "1"] = g[1][0]["ssr_ftest"][1]
        R["granger"].append(row)
    R["granger"] = pd.DataFrame(R["granger"])
    R["meta"]["weeks"] = len(wk)

    # --- walk-forward: refit monthly using only outcomes already known, predict the next month
    h = 21
    models = {
        "Breadth score": ["score"], "% above 50dma": ["pct50"], "Net 25% quarter": ["net25q"],
        "Net 13% / 34 days": ["net13"], "New-high share": ["nh_pct5"],
        "QQQ momentum only (no breadth)": ["p63", "p252"],
        "Breadth + QQQ momentum": ["score", "pct50", "net25q", "net_hl", "p63", "p252"],
    }
    first_test = df.index[0] + pd.DateOffset(years=3)
    months = [df.index[df.index.searchsorted(m)] for m in df.loc[first_test:].resample("MS").first().index]
    preds = {k: pd.Series(np.nan, df.index) for k in list(models) + ["mean"]}
    for i, t in enumerate(months):
        pos = df.index.get_loc(t)
        train = df.iloc[: max(pos - h, 0)]
        end = months[i + 1] if i + 1 < len(months) else df.index[-1] + pd.Timedelta(days=1)
        test = df.loc[(df.index >= t) & (df.index < end)]
        preds["mean"].loc[test.index] = train[f"f{h}"].mean()
        for k, cols in models.items():
            tr = train[cols + [f"f{h}"]].dropna()
            if len(tr) >= 250:
                m = sm.OLS(tr[f"f{h}"], sm.add_constant(tr[cols])).fit()
                preds[k].loc[test.index] = m.predict(sm.add_constant(test[cols], has_constant="add"))
    y = df[f"f{h}"]
    ok = y.notna()
    for k in preds:
        ok &= preds[k].notna()
    sse0 = ((y[ok] - preds["mean"][ok]) ** 2).sum()
    for k in models:
        e = y[ok] - preds[k][ok]
        R["oos"].append({"model": k, "r2": 1 - (e ** 2).sum() / sse0,
                         "hit": (np.sign(preds[k][ok] - preds["mean"][ok]) == np.sign(y[ok] - preds["mean"][ok])).mean()})
    R["oos"] = pd.DataFrame(R["oos"])
    R["meta"]["oos_start"] = y[ok].index[0]

    # --- timing rules: decided at the close, first return earned the NEXT day (1-day execution delay)
    tests = df.index >= first_test
    rules = {
        "Buy & hold QQQ": pd.Series(True, df.index),
        "In QQQ only when score >= 40": bb.score >= 40,
        "In QQQ only when score >= 50": bb.score >= 50,
        "Out only when score < 20": bb.score >= 20,
        "In only when >= 40% of stocks above 50dma": bb.pct50 >= 0.40,
        "Out when primary trend weak (down-25% qtr > up-25%)": bb.dn25q <= bb.up25q,
        "In only when walk-forward model is above average": preds["Breadth + QQQ momentum"] >= preds["mean"],
        "Price only: in when QQQ above its 200-day avg": df.qqq > df.qqq.rolling(200).mean(),
    }
    eq = {}
    for k, pos in rules.items():
        held = pos.astype(float).shift(2).fillna(0)
        r = (held * dret)[tests]
        R["strategies"].append({"rule": k, **perf(r), "invested": held[tests].mean(),
                                "switches": int((held[tests].diff().abs() > 0).sum())})
        eq[k] = (1 + r.fillna(0)).cumprod()
    R["strategies"] = pd.DataFrame(R["strategies"])
    R["equity"] = pd.DataFrame(eq)
    R["meta"]["strat_start"] = df.index[tests][0]

    # --- today and the closest historical analogs
    last = df.index[-1]
    R["now"] = {k: {"v": df[k].iloc[-1], "pct": (df[k].dropna() < df[k].iloc[-1]).mean()} for k in FEATURES}
    R["now_raw"] = b.loc[last]
    zz = df[["score", "pct50", "net25q", "net13"]].dropna()
    zz = (zz - zz.mean()) / zz.std()
    dist = np.sqrt(((zz - zz.iloc[-1]) ** 2).sum(axis=1))
    dist = dist[dist.index <= last - pd.Timedelta(days=100)]
    near = dist.nsmallest(300).sort_index()
    picks = decluster(pd.Series(True, near.index).reindex(df.index, fill_value=False))
    picks = sorted(picks, key=lambda d: dist[d])[:20]
    R["analogs"] = df.loc[sorted(picks), ["score", "pct50", "net25q", "net13", "s21", "s63"]]
    R["df"] = df
    return R

# ============================================================== 4. report
BG, PANEL, INK, MUTED, GRID = "#151a21", "#1c222b", "#e6edf3", "#8b949e", "#2a313c"
S1, S2, S3, S4 = "#3987e5", "#d95926", "#199e70", "#c98500"     # validated categorical order
GOOD, WARN, BAD = "#3f9f4c", "#d7a53a", "#d0463f"
UI = "Inter,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
MONO = "'JetBrains Mono',Consolas,'Liberation Mono',monospace"

pc = lambda v, d=1: "–" if pd.isna(v) else f"{v * 100:+.{d}f}%"
pp = lambda v: "–" if pd.isna(v) else f"{v * 100:.0f}%"
f2 = lambda v: "–" if pd.isna(v) else f"{v:+.2f}"

def tag(text, color):
    return (f'<span style="display:inline-block;padding:1px 8px;border-radius:9px;font-size:11.5px;'
            f'font-weight:600;border:1px solid {color};color:{color};white-space:nowrap">{text}</span>')

def table(headers, rows, left=(0,)):
    th = "".join(f'<th style="padding:6px 9px;text-align:{"left" if i in left else "right"};color:{MUTED};'
                 f'font-weight:600;border-bottom:1px solid {GRID};font-size:12px">{h}</th>' for i, h in enumerate(headers))
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f'<td style="padding:6px 9px;text-align:{"left" if i in left else "right"};border-bottom:1px solid {GRID};'
            f'{"" if i in left else "font-family:" + MONO + ";"}font-size:12.5px">{c}</td>' for i, c in enumerate(r)) + "</tr>"
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;color:{INK}">'
            f'<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>')

def section(title, sub, inner):
    return (f'<div style="background:{PANEL};border-radius:10px;padding:16px 18px;margin:12px 0">'
            f'<div style="font-size:16px;font-weight:600;margin-bottom:2px">{title}</div>'
            f'<div style="font-size:12.5px;color:{MUTED};margin-bottom:12px;line-height:1.45">{sub}</div>{inner}</div>')

def ret_cell(v, base):
    if pd.isna(v):
        return "–"
    col = GOOD if v > base + 0.005 else BAD if v < base - 0.005 else INK
    return f'<span style="color:{col}">{pc(v)}</span>'

def headline(R):
    sc, ev, st, oos, strat, gr = R["scorecard"], R["events"], R["states"], R["oos"], R["strategies"], R["granger"]
    out = []
    real = sc[sc.verdict == "Real edge"]
    weak = sc[sc.verdict == "Weak / unreliable"]
    if len(real):
        items = "; ".join(f"<b>{r.label}</b> ({'bullish when high' if r[f'ic{r.best_h}'] > 0 else 'bearish when high'}"
                          f", {r.way.split(',')[0]})" for _, r in real.iterrows())
        out.append(f"{len(real)} of {len(sc)} indicators showed a statistically solid link to QQQ's return over the "
                   f"following month or quarter that held in both halves of the decade and survived controlling for QQQ's "
                   f"own trend: {items}.")
    else:
        out.append(f"None of the {len(sc)} indicators showed a link to QQQ's future return that was both statistically "
                   f"solid and consistent across both halves of the decade, once QQQ's own trend is accounted for.")
    if len(weak):
        out.append(f"{len(weak)} more showed weak or inconsistent evidence: " + ", ".join(weak.label) + ".")
    good_oos = oos[oos.r2 > 0]
    mom = oos.set_index("model").r2.get("QQQ momentum only (no breadth)", np.nan)
    if len(good_oos):
        bst = good_oos.sort_values("r2").iloc[-1]
        out.append(f"Out of sample (models refit monthly, never seeing the future), the best breadth model — "
                   f"<b>{bst.model}</b> — explained {bst.r2 * 100:.1f}% more of next-month QQQ variation than simply assuming "
                   f"the average return (momentum-only model: {mom * 100:+.1f}%). Monthly stock returns are noisy, so "
                   f"even 1-3% is considered meaningful; below that treat it as noise.")
    else:
        out.append("Out of sample (models refit monthly, never seeing the future), <b>no breadth model beat simply "
                   "assuming QQQ's average return</b> for predicting the next month. That is the strictest test here.")
    bh = strat.iloc[0]
    alt = strat.iloc[1:-1]
    better_dd = alt[alt.maxdd > bh.maxdd + 0.05]
    better_sh = alt[alt.sharpe > bh.sharpe + 0.05]
    beat_ret = alt[alt.cagr > bh.cagr]
    s = (f"As trading rules (with a 1-day delay), buy-and-hold QQQ returned {bh.cagr * 100:.1f}%/yr with a "
         f"{bh.maxdd * 100:.0f}% worst drawdown. ")
    if len(beat_ret):
        s += f"{len(beat_ret)} breadth rule(s) beat it on return: " + ", ".join(beat_ret.rule) + ". "
    else:
        s += "No breadth rule beat it on total return. "
    if len(better_dd):
        s += f"{len(better_dd)} cut the worst drawdown by 5+ points"
        s += (f" and {len(better_sh)} improved risk-adjusted return (Sharpe)." if len(better_sh) else
              ", but none improved risk-adjusted return (Sharpe) — they mostly just held less stock.")
    out.append(s)
    b2q, q2b = (gr.b2q < 0.05).sum(), (gr.q2b < 0.05).sum()
    out.append(f"Direction of cause: in weekly data, breadth changes helped forecast QQQ in {b2q} of {len(gr)} tests, "
               f"while QQQ's own moves helped forecast breadth in {q2b} of {len(gr)}. "
               + ("Breadth mostly follows price rather than leading it." if q2b > b2q else
                  "Breadth leads price at least as often as it follows it." if b2q >= q2b and b2q > 0 else ""))
    vol_rows = st[st.vol_on > st.vol_off * 1.25]
    if len(vol_rows):
        out.append("Where breadth clearly helps is <b>risk</b>: when " + "; ".join(vol_rows.state.str.lower()) +
                   f", QQQ's volatility over the next month averaged {vol_rows.vol_on.mean() * 100:.0f}% annualized vs "
                   f"{vol_rows.vol_off.mean() * 100:.0f}% otherwise — bigger swings in both directions.")
    return out

def render(R):
    m, sc, now = R["meta"], R["scorecard"], R["now"]
    base = R["base"]
    parts = []
    # header
    raw = R["now_raw"]
    on_ev = R["events"][R["events"].on_now]
    on_st = R["states"][R["states"].on_now]
    an = R["analogs"]
    parts.append(f'''
<div style="font-size:20px;font-weight:700">Does market breadth predict QQQ?</div>
<div style="color:{MUTED};font-size:13px;margin:4px 0 10px">
  {m["start"]:%b %d, %Y} to {m["end"]:%b %d, %Y} · {m["days"]:,} trading days · breadth from current S&amp;P 1500
  members ({m["u0"]:,} eligible stocks at the start, {m["u1"]:,} now) · QQQ returned {m["qqq_cagr"] * 100:.1f}%/yr including dividends</div>''')
    parts.append(section("Bottom line", "Generated from the results below, not written in advance.",
                         "<ul style='margin:0;padding-left:18px;line-height:1.6;font-size:13.5px'>" +
                         "".join(f"<li style='margin:4px 0'>{t}</li>" for t in headline(R)) + "</ul>"))
    # today
    ev_txt = ", ".join(on_ev.event) if len(on_ev) else "none"
    st_txt = ", ".join(on_st.state) if len(on_st) else "none"
    rows = []
    for k in ["score", "pct50", "net25q", "net13", "r10", "nh_pct5", "adv5", "ext10_f"]:
        lab = FEATURES[k][0]
        v = now[k]["v"]
        disp = f"{v:.0f}" if k == "score" else f"{np.exp(v):.2f}" if k in ("r5", "r10") else f"{v * 100:.1f}%"
        rows.append([lab, disp, pp(now[k]["pct"]), sc.loc[k, "verdict"]])
    parts.append(section(
        f"Where things stand on {m['end']:%b %d, %Y}",
        f"Percentile = share of the last 10 years with a lower reading. Signals firing today: <b>{ev_txt}</b>. "
        f"Conditions in force: <b>{st_txt}</b>.",
        table(["Indicator", "Today", "10-yr percentile", "Predictive record"], rows) +
        f'<div style="font-size:13px;margin-top:12px;line-height:1.5">The {len(an)} most similar past setups '
        f'(closest score, % above 50dma, net 25% and net 13% readings, at least 20 days apart) were followed by an average QQQ move of '
        f'<b>{pc(an.s21.mean())}</b> over the next month ({pp((an.s21.dropna() > 0).mean())} positive) and '
        f'<b>{pc(an.s63.mean())}</b> over the next quarter ({pp((an.s63.dropna() > 0).mean())} positive). '
        f'Any-day averages: {pc(base[21]["m"])} and {pc(base[63]["m"])} ({pp(base[21]["p"])} / {pp(base[63]["p"])} positive).</div>'))
    # scorecard
    col = {"Real edge": GOOD, "Weak / unreliable": WARN, "No edge": MUTED, "Too rare to test": MUTED}
    rows = []
    for k, r in sc.iterrows():
        rows.append([r.label, f2(r.ic5), f2(r.ic21), f2(r.ic63), f2(r.tc21), f2(r.tc63),
                     f"{f2(r[f'ica{r.best_h}'])} / {f2(r[f'icb{r.best_h}'])}", tag(r.verdict, col[r.verdict]),
                     f'<span style="font-family:{UI};color:{MUTED};font-size:12px">{r.plain if r.verdict in ("Real edge", "Weak / unreliable") else ""}</span>'])
    parts.append(section(
        "Indicator scorecard",
        "IC = rank correlation between today's reading and QQQ's return over the next 1 week / 1 month / 1 quarter "
        "(0 = no relationship; in markets |IC| of 0.05-0.10 is useful, above 0.15 is rare). t = Newey-West t-statistic "
        "after also controlling for QQQ's own 1-, 3- and 12-month return and volatility, so it only counts what breadth adds "
        "beyond price (|t| ≥ 2 ≈ statistically significant). Halves = IC in the first vs. second half of the decade. "
        "<b>Real edge</b> needs |t| ≥ 2 and the same sign in both halves. With 17 indicators × 4 horizons, expect a few "
        "false positives by chance.",
        table(["Indicator", "IC 1wk", "IC 1mo", "IC 1qtr", "t 1mo", "t 1qtr", "Halves", "Verdict", "What it means"], rows, left=(0, 8))))
    # events
    ev = R["events"]
    rows = []
    for _, r in ev.iterrows():
        rows.append([r.event + (" " + tag("ON NOW", WARN) if r.on_now else ""), r.n,
                     ret_cell(r.m5, base[5]["m"]), ret_cell(r.m21, base[21]["m"]), ret_cell(r.m63, base[63]["m"]),
                     pp(r.p21), pp(r.p63), "–" if pd.isna(r.pv63) else f"{r.pv63:.2f}",
                     "–" if pd.isna(r["last"]) else f"{r['last']:%Y-%m-%d}"])
    rows.append([f'<span style="color:{MUTED}">Any day (baseline)</span>', "", pc(base[5]["m"]), pc(base[21]["m"]),
                 pc(base[63]["m"]), pp(base[21]["p"]), pp(base[63]["p"]), "", ""])
    parts.append(section(
        "What happened after each dashboard signal",
        "Average QQQ return after the first day of each episode (repeats within 20 trading days ignored, so episodes don't "
        "double-count). Green/red = better/worse than any-day baseline. p = chance random dates would give a 3-month "
        "average this far from normal (below 0.05 is notable; with few episodes treat everything as anecdotal).",
        table(["Signal", "Episodes", "Next 1wk", "Next 1mo", "Next 1qtr", "% up 1mo", "% up 1qtr", "p (1qtr)", "Last seen"], rows)))
    # states
    st = R["states"]
    rows = [[r.state + (" " + tag("NOW", WARN) if r.on_now else ""), pp(r.share),
             pc(r.on21), pc(r.off21), f2(r.t21), pc(r.on63), pc(r.off63), f2(r.t63),
             f"{r.vol_on * 100:.0f}% vs {r.vol_off * 100:.0f}%"] for _, r in st.iterrows()]
    parts.append(section(
        "Regimes: QQQ while a condition is on vs. off",
        "Average forward QQQ return on days the condition held vs. didn't, with Newey-West t for the difference. "
        "The last column is QQQ's realized volatility over the following month — breadth often says more about risk than direction.",
        table(["Condition", "% of days", "1mo on", "1mo off", "t", "1qtr on", "1qtr off", "t", "Next-month vol"], rows)))
    # granger
    g = R["granger"]
    fmt = lambda p: f'<span style="color:{GOOD if p < 0.05 else INK}">{p:.3f}</span>'
    rows = [[r["var"], fmt(r.b2q1), fmt(r.b2q), fmt(r.q2b1), fmt(r.q2b)] for _, r in g.iterrows()]
    parts.append(section(
        "Cause and effect: who leads whom? (Granger tests, weekly)",
        f"A Granger test asks whether last week's breadth improves a forecast of this week's QQQ beyond QQQ's own past "
        f"(and the reverse). p < 0.05 (green) = helps forecast. This is <i>predictive</i> causality only: breadth and QQQ "
        f"both respond to the same news, so no statistic here can prove one causes the other. {m['weeks']} weeks tested.",
        table(["Breadth measure", "Breadth → QQQ (1 wk lag)", "Breadth → QQQ (best of 1/2/4)",
               "QQQ → breadth (1 wk)", "QQQ → breadth (best of 1/2/4)"], rows)))
    # oos + strategies
    o = R["oos"]
    rows = [[r.model, f'<span style="color:{GOOD if r.r2 > 0 else BAD}">{r.r2 * 100:+.2f}%</span>', pp(r.hit)] for _, r in o.iterrows()]
    s = R["strategies"]
    srows = [[r.rule, f"{r.cagr * 100:.1f}%", f"{r.vol * 100:.1f}%", f"{r.sharpe:.2f}", f"{r.maxdd * 100:.0f}%",
              pp(r.invested), r.switches] for _, r in s.iterrows()]
    parts.append(section(
        "The no-hindsight test",
        f"From {m['oos_start']:%b %Y}, each model is refit every month using only data whose outcome was already known, then "
        f"predicts the next month's QQQ return. Out-of-sample R² above 0 means it beat just assuming the historical average. "
        f"Direction hit = how often it correctly called above- vs below-average months.",
        table(["Model (predicting next-month QQQ)", "Out-of-sample R²", "Direction hit"], rows) +
        f'<div style="height:14px"></div><div style="font-size:12.5px;color:{MUTED};margin-bottom:6px">Simple rules, '
        f'{m["strat_start"]:%b %Y} onward. Signal read at the close, position taken at the NEXT close; cash earns 0%.</div>' +
        table(["Rule", "Return / yr", "Volatility", "Sharpe", "Worst drawdown", "Time invested", "Trades"], srows)))
    parts.append(section("Caveats", "", f'''<ul style="margin:0;padding-left:18px;font-size:13px;line-height:1.55">
      <li><b>Survivorship bias:</b> breadth is built from <i>today's</i> S&amp;P 1500. Stocks that were dropped or went bust
          are missing, so past breadth looks healthier than it was, especially early in the decade.</li>
      <li><b>Mismatch:</b> breadth covers ~1,500 equal-counted stocks; QQQ is 100 cap-weighted mega-caps. Periods where a few
          giants carry the index (narrow markets) weaken any breadth-to-QQQ link.</li>
      <li><b>Small samples:</b> 10 years holds only ~40 non-overlapping quarters and a handful of crashes (2018 Q4, 2020, 2022,
          2025). Rare signals (washed out, thrust) have too few episodes to be conclusive.</li>
      <li><b>Correlation ≠ causation:</b> the dashboard measures the same market QQQ trades in. Predictive power, where it
          exists, reflects momentum and mean-reversion in investor behavior, not breadth causing prices.</li></ul>'''))
    return (f'<div style="background:{BG};color:{INK};font-family:{UI};padding:18px 20px;border-radius:10px">'
            + "".join(parts) + "</div>")

def charts(R):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    lay = dict(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family=UI, color=INK, size=12),
               margin=dict(l=60, r=30, t=60, b=40), hovermode="x unified",
               legend=dict(orientation="h", y=1.1, x=0))
    df = R["df"]
    # 1. QQQ and the breadth score, stacked on separate panels (no dual axis)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.07,
                        subplot_titles=("QQQ (dividend-adjusted)", "Breadth score"))
    fig.add_trace(go.Scatter(x=df.index, y=df.qqq, name="QQQ", line=dict(color=S1, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.score, name="Breadth score", line=dict(color=S4, width=1.5)), row=2, col=1)
    for y0, y1, c in [(60, 100, GOOD), (1, 40, BAD)]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=c, opacity=0.10, line_width=0, row=2, col=1)
    fig.update_yaxes(type="log", row=1, col=1, gridcolor=GRID)
    fig.update_yaxes(range=[0, 100], row=2, col=1, gridcolor=GRID)
    fig.update_layout(height=560, showlegend=False, **{k: v for k, v in lay.items() if k != "legend"})
    fig.show()
    # 2. quintiles: next-quarter QQQ return by indicator bucket
    fig = go.Figure()
    for k, c in [("score", S1), ("pct50", S2), ("net25q", S3), ("net13", S4)]:
        qd = R["quint"][k]
        fig.add_trace(go.Bar(x=[f"Q{i}" for i in qd.q], y=qd.m63 * 100, name=FEATURES[k][0], marker_color=c,
                             marker_line=dict(color=BG, width=2),
                             customdata=np.c_[qd.lo, qd.hi, qd.p63 * 100],
                             hovertemplate="%{y:.1f}% avg · %{customdata[2]:.0f}% positive<extra>" + FEATURES[k][0] + "</extra>"))
    fig.add_hline(y=R["base"][63]["m"] * 100, line=dict(color=MUTED, dash="dash", width=1),
                  annotation_text="any-day average", annotation_font_color=MUTED)
    fig.update_layout(title="Average QQQ return over the next quarter, by indicator quintile (Q1 = lowest 20% of readings)",
                      barmode="group", bargap=0.25, height=420, yaxis=dict(title="%", gridcolor=GRID), **lay)
    fig.update_layout(hovermode="closest")
    fig.show()
    # 3. growth of $1 for a few rules
    eq = R["equity"]
    keep = ["Buy & hold QQQ", "In QQQ only when score >= 40", "Out when primary trend weak (down-25% qtr > up-25%)",
            "Price only: in when QQQ above its 200-day avg"]
    fig = go.Figure()
    for k, c in zip(keep, [S1, S2, S3, S4]):
        fig.add_trace(go.Scatter(x=eq.index, y=eq[k], name=k, line=dict(color=c, width=2)))
    fig.update_layout(title="Growth of $1: buy & hold vs. breadth rules (log scale, 1-day execution delay)",
                      height=440, yaxis=dict(type="log", gridcolor=GRID), **lay)
    fig.show()

def paste_back(R):
    m, sc = R["meta"], R["scorecard"]
    L = [f"BREADTH vs QQQ {m['start']:%Y-%m-%d}..{m['end']:%Y-%m-%d} days={m['days']} univ={m['u0']}->{m['u1']} "
         f"qqq_cagr={m['qqq_cagr']:.3f} sameday_corr={m['same_day']:.2f} nextday_corr={m['next_day']:.3f}",
         "base " + " ".join(f"h{h}:{v['m']:.4f}/{v['p']:.2f}" for h, v in R["base"].items()),
         "SCORECARD key ic5 ic21 ic63 t21 t63 tc21 tc63 halves verdict"]
    for k, r in sc.iterrows():
        L.append(f"{k} {r.ic5:.3f} {r.ic21:.3f} {r.ic63:.3f} {r.t21:.2f} {r.t63:.2f} {r.tc21:.2f} {r.tc63:.2f} "
                 f"{r[f'ica{r.best_h}']:.3f}/{r[f'icb{r.best_h}']:.3f}@{r.best_h} {r.verdict}")
    L.append("QUINTILES m21/m63 Q1..Q5")
    for k in ["score", "pct50", "net25q", "net13", "r10", "nh_pct5", "dn4_f", "ext10_f"]:
        qd = R["quint"][k]
        L.append(f"{k} " + " ".join(f"{a:.3f}/{b:.3f}" for a, b in zip(qd.m21, qd.m63)))
    L.append("EVENTS n m5 m21 m63 p21 p63 pv63 on_now last")
    for _, r in R["events"].iterrows():
        L.append(f"{r.event[:40]} | {r.n} {r.m5:.3f} {r.m21:.3f} {r.m63:.3f} {r.p21:.2f} {r.p63:.2f} {r.pv63:.2f} "
                 f"{int(r.on_now)} {'' if pd.isna(r['last']) else r['last'].strftime('%Y-%m-%d')}")
    L.append("STATES share on21 off21 t21 on63 off63 t63 vol_on vol_off now")
    for _, r in R["states"].iterrows():
        L.append(f"{r.state[:40]} | {r.share:.2f} {r.on21:.3f} {r.off21:.3f} {r.t21:.2f} {r.on63:.3f} {r.off63:.3f} "
                 f"{r.t63:.2f} {r.vol_on:.2f} {r.vol_off:.2f} {int(r.on_now)}")
    L.append("GRANGER b2q_1 b2q_best q2b_1 q2b_best")
    for _, r in R["granger"].iterrows():
        L.append(f"{r['var']} | {r.b2q1:.3f} {r.b2q:.3f} {r.q2b1:.3f} {r.q2b:.3f}")
    L.append(f"OOS from {m['oos_start']:%Y-%m}: " + "; ".join(f"{r.model}={r.r2:.4f}/{r.hit:.2f}" for _, r in R["oos"].iterrows()))
    L.append("STRATS cagr vol sharpe maxdd invested switches")
    for _, r in R["strategies"].iterrows():
        L.append(f"{r.rule} | {r.cagr:.3f} {r.vol:.3f} {r.sharpe:.2f} {r.maxdd:.3f} {r.invested:.2f} {r.switches}")
    raw = R["now_raw"]
    L.append("NOW " + " ".join(f"{k}={raw[k]:.3g}" for k in ["score", "up4", "dn4", "r5", "r10", "up25q", "dn25q", "up25m",
                                                          "dn25m", "up13", "dn13", "ext10", "pct50", "adv_pct", "nh", "nl", "universe"]))
    L.append("NOW_PCTILE " + " ".join(f"{k}={v['pct']:.2f}" for k, v in R["now"].items()))
    an = R["analogs"]
    L.append(f"ANALOGS n={len(an)} m21={an.s21.mean():.3f} p21={(an.s21.dropna() > 0).mean():.2f} "
             f"m63={an.s63.mean():.3f} p63={(an.s63.dropna() > 0).mean():.2f} dates=" +
             ",".join(f"{d:%Y-%m-%d}" for d in an.index))
    return "\n".join(L)

# ============================================================== run
px_data = load_data()
breadth, _ = add_score(compute_breadth(px_data))
qqq = px_data["QQQ"].dropna()
print(f"Breadth history: {breadth.index[0]:%Y-%m-%d} to {breadth.index[-1]:%Y-%m-%d}. Running tests...")
R = analyze(breadth, qqq)
report_html = render(R)
display(HTML(report_html))
charts(R)
with open("breadth_vs_qqq_report.html", "w") as f:
    f.write(f"<html><body style='background:{BG};margin:0;padding:16px'>{report_html}</body></html>")
breadth.to_csv("breadth_history_full.csv")
print("\nSaved breadth_vs_qqq_report.html and breadth_history_full.csv (Files panel on the left).")
print("\n" + "=" * 30 + " COPY EVERYTHING BELOW THIS LINE AND PASTE IT BACK TO CLAUDE " + "=" * 30)
print(paste_back(R))
