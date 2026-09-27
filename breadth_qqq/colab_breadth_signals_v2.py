# Breadth signals v2: contrarian labels + out-of-sample test.  One Colab cell.
# The 2016-2026 study showed the dashboard reads breadth backwards: the weakest readings came before the best QQQ
# returns. This version relabels the score zones and signals accordingly, then tests them on 2006-2016 -- a decade
# the first study never looked at (includes the 2008 crash) -- as well as on 2016-2026.
# Paste this whole thing into ONE Colab cell (you can replace the previous cell) and press play.

!pip -q install -U yfinance     # plotly and statsmodels are already in Colab

EXTEND_HISTORY = True          # True: download from 2005 and test on 2006-2016 too (~10-15 min first run)
                               # False: 2016-2026 only, reuses the data already in memory from the last run
START_DATE = "2005-01-01" if EXTEND_HISTORY else "2014-06-01"
HOLDOUT = ("2006-06-01", "2016-09-26")     # never looked at when the new labels were chosen
DISCOVERY = ("2016-09-27", "2099-12-31")   # the decade the patterns were found in
HORIZONS = [5, 21, 63, 126]    # forward QQQ returns in trading days: 1 week, 1 month, 3 months, 6 months
HOLD_DAYS = 63                 # how long the strategy tests hold after a buy signal (3 months)
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 2e6
REQUIRE_VOLUME_CONFIRM = True
CACHE_FILE = f"breadth_prices_{START_DATE[:4]}.pkl"
FORCE_REDOWNLOAD = False

# New contrarian labels for the 1-100 score (the score itself is unchanged)
ZONES = [(0, 20, "Undervalued"), (20, 40, "Oversold"), (40, 60, "Neutral"), (60, 80, "Extended"), (80, 101, "Overbought")]
WASHED_OUT = 0.20              # % of stocks above their 50dma at or below this also counts as "Undervalued"

WEIGHTS = {
    "5-day ratio": 0.10, "10-day ratio": 0.10, "Quarter (±25%)": 0.15, "Month (±25%)": 0.10,
    "34-day (±13%)": 0.10, "% above 50dma": 0.15, "New highs vs lows": 0.10, "Advance/decline": 0.10,
    "50dma trend (5d)": 0.05, "S&P trend": 0.05,
}

import io, os, time, pickle, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from IPython.display import HTML, display
warnings.filterwarnings("ignore")
T0 = time.time()
def step(msg):
    print(f"[{(time.time() - T0) / 60:4.1f} min] {msg}", flush=True)
step("Libraries loaded.")
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
        step(f"  downloaded {min(i + chunk, len(tickers)):,} of {len(tickers):,} tickers")
        time.sleep(pause)
    raw = pd.concat(frames, axis=1)
    return raw.loc[:, ~raw.columns.duplicated()]

def load_data():
    key = (START_DATE, datetime.now().strftime("%Y-%m-%d"))
    if os.path.exists(CACHE_FILE) and not FORCE_REDOWNLOAD:
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        if cached.get("key") == key:
            step("Using prices cached earlier today (skipping the download).")
            return cached["px"]
    step("Step 1 of 4: getting the S&P 1500 ticker list from Wikipedia...")
    tickers = sp1500()
    step(f"Step 1 of 4: downloading daily bars since {START_DATE} for {len(tickers):,} stocks "
         f"(about 5-15 minutes; a line prints every 100 tickers)...")
    raw = download_prices(tickers)
    px = {f: raw[f] for f in ["Close", "High", "Low", "Volume"]}
    good = px["Close"].notna().sum() >= 60
    px = {f: v.loc[:, good[good].index] for f, v in px.items()}
    px["SPX"] = _one("^GSPC")
    px["QQQ"] = _one("QQQ")          # dividend-adjusted, so returns include dividends
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"key": key, "px": px}, f)
    step(f"Got usable data for {good.sum():,} tickers.")
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


# ============================================================== 3. signals with the new labels
def zone_of(score):
    for lo, hi, name in ZONES:
        if lo <= score < hi:
            return name
    return "n/a"

def decluster(mask, gap=20):
    """First day of each episode; repeats within `gap` trading days are ignored."""
    out, last = pd.Series(False, mask.index), -10**9
    for i, v in enumerate(mask.values):
        if v and i - last > gap:
            out.iloc[i] = True
            last = i
    return out

def build(b, qqq):
    df = b.join(qqq.rename("qqq"), how="inner").copy()
    lp = np.log(df.qqq)
    for h in HORIZONS:
        df[f"r{h}"] = np.exp(lp.shift(-h) - lp) - 1
    fmin = df.qqq[::-1].rolling(63, min_periods=1).min()[::-1].shift(-1)     # lowest close in the next 3 months
    df["dd63"] = fmin / df.qqq - 1
    df["vol21"] = lp.diff().rolling(21).std().shift(-21) * np.sqrt(252)       # realized vol over the next month
    df["sma200"] = df.qqq.rolling(200).mean()
    df["zone"] = df.score.apply(lambda s: zone_of(s) if pd.notna(s) else "n/a")
    ema_adv = df.adv_pct.ewm(span=10, adjust=False).mean()
    raw = {
        "Undervalued (score < 20 or <=20% above 50dma)": (df.score < 20) | (df.pct50 <= WASHED_OUT),
        "  - score below 20": df.score < 20,
        "  - washed out (<=20% above 50dma)": df.pct50 <= WASHED_OUT,
        "Oversold (score drops below 40)": (df.score < 40) & (df.score.shift(1) >= 40),
        "Oversold (10 Day Ratio <= 0.5)": df.r10 <= 0.5,
        "Breadth thrust (Zweig)": (ema_adv > 0.615) & (ema_adv.rolling(10).min() < 0.40),
        "Overbought (score 80+)": df.score >= 80,
        "Divergence (S&P near high, <40% above 50dma)": (df.spx >= 0.97 * df.spx_hi252) & (df.pct50 < 0.40),
    }
    events = {k: decluster(v.fillna(False).astype(bool)) for k, v in raw.items()}
    return df, events, raw

def boot_p(sample, pool, n=20000):
    if len(sample) < 3:
        return np.nan
    sims = RNG.choice(pool, size=(n, len(sample))).mean(axis=1)
    return float((np.abs(sims - pool.mean()) >= abs(sample.mean() - pool.mean())).mean())

def perf(ret):
    ret = ret.dropna()
    eq = (1 + ret).cumprod()
    return {"cagr": eq.iloc[-1] ** (252 / len(ret)) - 1, "vol": ret.std() * np.sqrt(252),
            "sharpe": ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan,
            "maxdd": (eq / eq.cummax() - 1).min()}

def strategies(df, events):
    uv = events["Undervalued (score < 20 or <=20% above 50dma)"]
    os_ = events["Oversold (score drops below 40)"] | uv
    after = lambda ev, n: ev.astype(float).rolling(n, min_periods=1).max() > 0
    trend = df.qqq > df.sma200
    return {
        "Buy & hold QQQ": pd.Series(True, df.index),
        "Trend: in when QQQ above 200-day avg": trend,
        f"Trend + buy Undervalued signals (hold {HOLD_DAYS}d)": trend | after(uv, HOLD_DAYS),
        f"Trend + buy Oversold/Undervalued (hold {HOLD_DAYS}d)": trend | after(os_, HOLD_DAYS),
        f"Only Undervalued signals (hold {HOLD_DAYS}d, else cash)": after(uv, HOLD_DAYS),
    }

def analyze_period(df, events, raw, start, end):
    d = df.loc[start:end]
    R = {"start": d.index[0], "end": d.index[-1], "days": len(d),
         "base": {h: {"m": d[f"r{h}"].mean(), "p": (d[f"r{h}"].dropna() > 0).mean()} for h in HORIZONS},
         "base_dd": d.dd63.mean(), "base_vol": d.vol21.mean(), "u0": int(d.universe.iloc[0])}
    rows = []
    for _, _, name in ZONES:
        sel = d[d.zone == name]
        rows.append({"zone": name, "share": len(sel) / len(d), **{f"m{h}": sel[f"r{h}"].mean() for h in HORIZONS},
                     "p63": (sel.r63.dropna() > 0).mean() if len(sel) else np.nan,
                     "vol": sel.vol21.mean(), "dd": sel.dd63.mean()})
    R["zones"] = pd.DataFrame(rows)
    rows = []
    for name, ev in events.items():
        dates = ev.loc[start:end]
        dates = dates[dates].index
        row = {"signal": name, "n": len(dates), "dates": list(dates)}
        for h in HORIZONS:
            y = d.loc[dates, f"r{h}"].dropna()
            row[f"m{h}"] = y.mean() if len(y) else np.nan
            row[f"p{h}"] = (y > 0).mean() if len(y) else np.nan
            row[f"pv{h}"] = boot_p(y.values, d[f"r{h}"].dropna().values)
        row["dd"] = d.loc[dates, "dd63"].mean() if len(dates) else np.nan
        rows.append(row)
    R["signals"] = pd.DataFrame(rows)
    # robustness: is the Undervalued edge a knife-edge threshold?
    grid = []
    for kind, col, levels in [("score below", "score", [15, 20, 25, 30]), ("% above 50dma at or below", "pct50", [0.15, 0.20, 0.25, 0.30])]:
        for lv in levels:
            m = (df[col] < lv) if col == "score" else (df[col] <= lv)
            ev = decluster(m.fillna(False)).loc[start:end]
            y = d.loc[ev[ev].index, "r63"].dropna()
            grid.append({"rule": f"{kind} {lv:.0%}" if col == "pct50" else f"{kind} {lv}", "n": len(y),
                         "m63": y.mean() if len(y) else np.nan, "p63": (y > 0).mean() if len(y) else np.nan})
    R["grid"] = pd.DataFrame(grid)
    uvname = "Undervalued (score < 20 or <=20% above 50dma)"
    uvd = R["signals"].set_index("signal").loc[uvname, "dates"]
    R["episodes"] = d.loc[uvd, ["score", "pct50", "r21", "r63", "r126", "dd63"]]
    dret = df.qqq.pct_change()
    rows, eq = [], {}
    for k, pos in strategies(df, events).items():
        held = pos.astype(float).shift(2).fillna(0).loc[start:end]      # decided at the close, earning from the next close
        r = held * dret.loc[start:end]
        rows.append({"rule": k, **perf(r), "invested": held.mean(), "trades": int((held.diff().abs() > 0).sum())})
        eq[k] = (1 + r.fillna(0)).cumprod()
    R["strats"] = pd.DataFrame(rows)
    R["equity"] = pd.DataFrame(eq)
    return R

def verdict(h, dsc):
    """Compare a signal's 3-month result against each period's normal 3-month return."""
    if h is None or pd.isna(h.get("m63")) or h["n"] < 3:
        return "Too few episodes to test", "#8b949e"
    eh, ed = h["m63"] - h["b63"], dsc["m63"] - dsc["b63"]
    if np.sign(eh) != np.sign(ed):
        return "Did not hold up out of sample", "#d0463f"
    if h["pv63"] < 0.10:
        return "Confirmed out of sample", "#3f9f4c"
    return "Same direction, not significant", "#d7a53a"

# ============================================================== 4. report
BG, PANEL, INK, MUTED, GRID = "#151a21", "#1c222b", "#e6edf3", "#8b949e", "#2a313c"
S1, S2, S3, S4 = "#3987e5", "#d95926", "#199e70", "#c98500"
GOOD, WARN, BAD = "#3f9f4c", "#d7a53a", "#d0463f"
ZONE_COL = {"Undervalued": "#2e9642", "Oversold": "#6fbf73", "Neutral": "#8b949e", "Extended": "#d7a53a", "Overbought": "#d0463f"}
UI = "Inter,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
MONO = "'JetBrains Mono',Consolas,'Liberation Mono',monospace"
pc = lambda v, d=1: "–" if pd.isna(v) else f"{v * 100:+.{d}f}%"
pp = lambda v: "–" if pd.isna(v) else f"{v * 100:.0f}%"

def tag(text, color):
    return (f'<span style="display:inline-block;padding:1px 8px;border-radius:9px;font-size:11.5px;font-weight:600;'
            f'border:1px solid {color};color:{color};white-space:nowrap">{text}</span>')

def table(headers, rows, left=(0,)):
    th = "".join(f'<th style="padding:6px 9px;text-align:{"left" if i in left else "right"};color:{MUTED};font-weight:600;'
                 f'border-bottom:1px solid {GRID};font-size:12px">{h}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(
        f'<td style="padding:6px 9px;text-align:{"left" if i in left else "right"};border-bottom:1px solid {GRID};'
        f'{"" if i in left else "font-family:" + MONO + ";"}font-size:12.5px">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;color:{INK}">'
            f'<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>')

def section(title, sub, inner):
    return (f'<div style="background:{PANEL};border-radius:10px;padding:16px 18px;margin:12px 0">'
            f'<div style="font-size:16px;font-weight:600;margin-bottom:2px">{title}</div>'
            f'<div style="font-size:12.5px;color:{MUTED};margin-bottom:12px;line-height:1.45">{sub}</div>{inner}</div>')

def rc(v, base):
    if pd.isna(v):
        return "–"
    col = GOOD if v > base + 0.005 else BAD if v < base - 0.005 else INK
    return f'<span style="color:{col}">{pc(v)}</span>'

def combined_signals(P):
    """One row per signal with both periods side by side."""
    out = []
    for name in P["dsc"]["signals"].signal:
        row = {"signal": name}
        for key in P:
            s = P[key]["signals"].set_index("signal").loc[name]
            row[key] = {"n": s.n, "m63": s.m63, "p63": s.p63, "m126": s.m126, "pv63": s.pv63, "m21": s.m21, "dd": s.dd,
                        "b63": P[key]["base"][63]["m"], "b126": P[key]["base"][126]["m"]}
        out.append(row)
    return out

def headline(P, cs, now):
    L = []
    uv = next(r for r in cs if r["signal"].startswith("Undervalued"))
    d = uv["dsc"]
    for key, label in [("hold", "2006-2016 (never looked at)"), ("dsc", "2016-2026")]:
        if key not in uv:
            continue
        x = uv[key]
        L.append(f"<b>Undervalued signal, {label}:</b> {x['n']} episodes, QQQ averaged {pc(x['m63'])} over the next "
                 f"3 months ({pp(x['p63'])} positive) vs {pc(x['b63'])} normally, and {pc(x['m126'])} over 6 months "
                 f"vs {pc(x['b126'])}. Typical dip after the signal before the 3 months were up: {pc(x['dd'])}.")
    if "hold" in P:
        z = {k: P[k]["zones"].set_index("zone") for k in P}
        lo_hi = [(k, z[k].loc["Undervalued", "m63"] - z[k].loc["Overbought", "m63"]) for k in P]
        same = all(v > 0 for _, v in lo_hi if pd.notna(v))
        L.append("Across both decades, days in the <b>Undervalued</b> zone were followed by "
                 + ("better" if same else "not consistently better")
                 + " 3-month returns than days in the <b>Overbought</b> zone ("
                 + ", ".join(f"{'2006-16' if k == 'hold' else '2016-26'}: {pc(v)} gap" for k, v in lo_hi) + ").")
        verdicts = [(r["signal"].strip(), verdict(r["hold"], r["dsc"])[0]) for r in cs if not r["signal"].startswith("  -")]
        good = [s for s, v in verdicts if v.startswith("Confirmed")]
        bad = [s for s, v in verdicts if v.startswith("Did not")]
        L.append(f"Out-of-sample check: {len(good)} signal(s) confirmed on 2006-2016"
                 + (f" ({'; '.join(good)})" if good else "") + f"; {len(bad)} reversed"
                 + (f" ({'; '.join(bad)})" if bad else "") + ".")
    for key in P:
        s = P[key]["strats"].set_index("rule")
        bh, tr = s.iloc[0], s.iloc[1]
        best = s.iloc[2]
        per = "2006-16" if key == "hold" else "2016-26"
        L.append(f"<b>Trading it ({per}):</b> buy & hold {bh.cagr:.1%}/yr (worst drop {bh.maxdd:.0%}); 200-day trend "
                 f"{tr.cagr:.1%}/yr ({tr.maxdd:.0%}); trend + buying Undervalued signals {best.cagr:.1%}/yr "
                 f"({best.maxdd:.0%}, Sharpe {best.sharpe:.2f} vs {bh.sharpe:.2f} buy & hold).")
    L.append(f"<b>Today</b> the score is {now['score']:.0f}, which is the <b>{now['zone']}</b> zone, with "
             f"{now['pct50']:.0%} of stocks above their 50dma (Undervalued triggers at {WASHED_OUT:.0%} or a score under 20).")
    return L

def ladder(score):
    segs = "".join(f'<div style="flex:{hi - lo};background:{ZONE_COL[n]};opacity:.85;height:12px"></div>' for lo, hi, n in ZONES)
    labels = "".join(f'<div style="flex:{min(hi, 100) - lo};text-align:center;font-size:11px;color:{MUTED}">{n}<br>{lo}-{min(hi, 100)}</div>'
                     for lo, hi, n in ZONES)
    return (f'<div style="position:relative;margin:10px 0 4px"><div style="display:flex;gap:2px;border-radius:6px;overflow:hidden">{segs}</div>'
            f'<div style="position:absolute;left:calc({score:.0f}% - 2px);top:-5px;width:4px;height:22px;background:#fff;border-radius:2px"></div></div>'
            f'<div style="display:flex;gap:2px">{labels}</div>')

def render(P, cs, df, now):
    parts = []
    periods = [(k, "2006–2016<br><span style='font-weight:400'>out of sample</span>" if k == "hold" else "2016–2026<br><span style='font-weight:400'>where found</span>") for k in P]
    parts.append(f'<div style="font-size:20px;font-weight:700">Breadth signals, relabeled: do they predict QQQ?</div>'
                 f'<div style="color:{MUTED};font-size:13px;margin:4px 0 10px">Score zones and signals now read breadth as '
                 f'contrarian. Tested on ' + " and ".join(f"{P[k]['start']:%b %Y}–{P[k]['end']:%b %Y}" for k in P) +
                 '. Current S&amp;P 1500 members; QQQ dividend-adjusted.</div>')
    parts.append(section("Bottom line", "Generated from the results below.",
                         "<ul style='margin:0;padding-left:18px;line-height:1.6;font-size:13.5px'>" +
                         "".join(f"<li style='margin:4px 0'>{t}</li>" for t in headline(P, cs, now)) + "</ul>"))
    firing = [r["signal"].strip() for r in cs if now["events_now"].get(r["signal"])]
    parts.append(section(f"Today ({now['date']:%b %d, %Y}) with the new labels",
                         f"Signals active in the last 20 trading days: <b>{', '.join(firing) if firing else 'none'}</b>.",
                         f'<div style="display:flex;gap:28px;align-items:center;flex-wrap:wrap">'
                         f'<div><div style="font-size:13px;color:{MUTED}">Breadth score</div>'
                         f'<div style="font-size:52px;font-weight:700;font-family:{MONO};color:{ZONE_COL[now["zone"]]}">{now["score"]:.0f}</div>'
                         f'<div style="font-size:18px;font-weight:600;color:{ZONE_COL[now["zone"]]}">{now["zone"]}</div></div>'
                         f'<div style="flex:1;min-width:280px">{ladder(now["score"])}</div></div>'))
    # zones
    hdr = ["Zone"] + [f"{'06–16' if k == 'hold' else '16–26'} {c}" for k, _ in periods for c in ["% days", "3 mo", "6 mo", "% up 3 mo", "next-mo vol"]]
    rows = []
    for _, _, name in ZONES:
        r = [tag(name, ZONE_COL[name])]
        for k, _ in periods:
            z = P[k]["zones"].set_index("zone").loc[name]
            r += [pp(z.share), rc(z.m63, P[k]["base"][63]["m"]), rc(z.m126, P[k]["base"][126]["m"]), pp(z.p63),
                  "–" if pd.isna(z.vol) else f"{z.vol:.0%}"]
        rows.append(r)
    rows.append([f'<span style="color:{MUTED}">Any day</span>'] + sum(
        [["", pc(P[k]["base"][63]["m"]), pc(P[k]["base"][126]["m"]), pp(P[k]["base"][63]["p"]), f"{P[k]['base_vol']:.0%}"] for k, _ in periods], []))
    parts.append(section("Score zones with the new names",
                         "Average QQQ return over the next 3 and 6 months for every day spent in each zone. Green/red = better/worse "
                         "than that decade's normal. Next-month vol = QQQ's realized volatility over the following month.",
                         table(hdr, rows)))
    # signals
    hdr = ["Signal"] + sum([[f"{'06–16' if k == 'hold' else '16–26'} n", "1 mo", "3 mo", "% up", "p"] for k, _ in periods], []) + (["Verdict"] if "hold" in P else [])
    rows = []
    for r in cs:
        row = [r["signal"].replace("  - ", "&nbsp;&nbsp;↳ ")]
        for k, _ in periods:
            x = r[k]
            row += [x["n"], rc(x["m21"], P[k]["base"][21]["m"]), rc(x["m63"], x["b63"]), pp(x["p63"]),
                    "–" if pd.isna(x["pv63"]) else f"{x['pv63']:.2f}"]
        if "hold" in P:
            v, c = verdict(r["hold"], r["dsc"])
            row.append(tag(v, c))
        rows.append(row)
    parts.append(section("Signals: 2006–2016 (out of sample) next to 2016–2026" if "hold" in P else "Signals, 2016–2026",
                         "QQQ return after the first day of each episode (repeats within 20 trading days ignored). p = chance random days "
                         "would average this far from normal over 3 months (below 0.10 = notable). <b>Confirmed out of sample</b> = same "
                         "direction in both decades and p &lt; 0.10 in the decade the rules were not built on.",
                         table(hdr, rows, left=(0, len(hdr) - 1) if "hold" in P else (0,))))
    # robustness grid
    hdr = ["Threshold"] + sum([[f"{'06–16' if k == 'hold' else '16–26'} n", "3 mo", "% up"] for k, _ in periods], [])
    rules = P["dsc"]["grid"].rule
    rows = []
    for i, rule in enumerate(rules):
        row = [rule + (" ← used" if rule in ("score below 20", "% above 50dma at or below 20%") else "")]
        for k, _ in periods:
            g = P[k]["grid"].iloc[i]
            row += [g.n, rc(g.m63, P[k]["base"][63]["m"]), pp(g.p63)]
        rows.append(row)
    parts.append(section("Is it a lucky threshold?",
                         "The Undervalued edge recomputed with nearby cutoffs. If only the exact cutoff works, it is probably curve-fitting.",
                         table(hdr, rows)))
    # episodes
    for k, _ in periods:
        ep = P[k]["episodes"]
        rows = [[f"{d:%Y-%m-%d}", f"{r.score:.0f}", pp(r.pct50), pc(r.r21), pc(r.r63), pc(r.r126), pc(r.dd63)] for d, r in ep.iterrows()]
        parts.append(section(f"Every Undervalued signal, {'2006–2016' if k == 'hold' else '2016–2026'}",
                             "Worst dip = lowest QQQ close in the 3 months after the signal, vs the signal-day close.",
                             table(["Date", "Score", "% >50dma", "Next 1 mo", "Next 3 mo", "Next 6 mo", "Worst dip"], rows)))
    # strategies
    for k, _ in periods:
        s = P[k]["strats"]
        rows = [[r.rule, f"{r.cagr:.1%}", f"{r.vol:.1%}", f"{r.sharpe:.2f}", f"{r.maxdd:.0%}", pp(r.invested), r.trades] for _, r in s.iterrows()]
        parts.append(section(f"Trading rules, {P[k]['start']:%b %Y}–{P[k]['end']:%b %Y}",
                             "Signal read at the close, position taken at the next close, cash earns 0%.",
                             table(["Rule", "Return/yr", "Volatility", "Sharpe", "Worst drawdown", "Time invested", "Trades"], rows)))
    parts.append(section("Caveats", "", f'''<ul style="margin:0;padding-left:18px;font-size:13px;line-height:1.55">
      <li><b>Survivorship bias</b> is worse in 2006–2016: only stocks that are in the S&amp;P 1500 <i>today</i> are counted, so the
          2008 losers that were later dropped are missing and breadth looks healthier than it was.</li>
      <li><b>Few independent events:</b> "Undervalued" episodes cluster around a handful of selloffs per decade. Treat the
          percentages as rough.</li>
      <li><b>Labels are not valuations:</b> "Undervalued" here means breadth is washed out, not that stocks are cheap on earnings.</li></ul>'''))
    return f'<div style="background:{BG};color:{INK};font-family:{UI};padding:18px 20px;border-radius:10px">' + "".join(parts) + "</div>"

def charts(P, df, events):
    import plotly.graph_objects as go
    lay = dict(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family=UI, color=INK, size=12),
               margin=dict(l=60, r=30, t=70, b=40), legend=dict(orientation="h", y=1.1, x=0))
    start = min(P[k]["start"] for k in P)
    d = df.loc[start:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d.qqq, name="QQQ", line=dict(color="#9aa4b2", width=1.4), hoverinfo="skip"))
    for name, col, sym in [("Undervalued (score < 20 or <=20% above 50dma)", S3, "triangle-up"),
                           ("Oversold (score drops below 40)", S1, "circle"),
                           ("Overbought (score 80+)", S2, "triangle-down"),
                           ("Divergence (S&P near high, <40% above 50dma)", S4, "diamond")]:
        ev = events[name].loc[start:]
        dates = ev[ev].index
        fig.add_trace(go.Scatter(x=dates, y=d.loc[dates, "qqq"], mode="markers", name=name.split(" (")[0],
                                 marker=dict(color=col, size=9, symbol=sym, line=dict(color=BG, width=1.5)),
                                 customdata=np.c_[d.loc[dates, "r63"] * 100],
                                 hovertemplate="%{x|%Y-%m-%d}<br>next 3 mo: %{customdata[0]:+.1f}%<extra>" + name.split(" (")[0] + "</extra>"))
    if "hold" in P:
        fig.add_vrect(x0=P["hold"]["start"], x1=P["hold"]["end"], fillcolor="#3987e5", opacity=0.06, line_width=0,
                      annotation_text="out of sample", annotation_position="top left", annotation_font_color=MUTED)
    fig.update_layout(title="QQQ (log scale) with the relabeled breadth signals", height=480,
                      yaxis=dict(type="log", gridcolor=GRID), hovermode="closest", **lay)
    fig.show()
    fig = go.Figure()
    for k, col in [("hold", S1), ("dsc", S2)]:
        if k not in P:
            continue
        z = P[k]["zones"]
        fig.add_trace(go.Bar(x=z.zone, y=(z.m63 - P[k]["base"][63]["m"]) * 100, name="2006–2016 (out of sample)" if k == "hold" else "2016–2026",
                             marker_color=col, marker_line=dict(color=BG, width=2),
                             hovertemplate="%{x}: %{y:+.1f} pts vs normal<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(title="3-month QQQ return by score zone, compared with that decade's normal (percentage points)",
                      barmode="group", bargap=0.3, height=400, yaxis=dict(gridcolor=GRID, title="pts vs normal"), hovermode="closest", **lay)
    fig.show()

def paste_back(P, cs, now):
    L = [f"BREADTH SIGNALS v2  today={now['date']:%Y-%m-%d} score={now['score']:.0f} zone={now['zone']} pct50={now['pct50']:.3f} "
         f"firing={[r['signal'].strip() for r in cs if now['events_now'].get(r['signal'])]}"]
    for k in P:
        p = P[k]
        L.append(f"PERIOD {k} {p['start']:%Y-%m-%d}..{p['end']:%Y-%m-%d} univ0={p['u0']} base21={p['base'][21]['m']:.4f} "
                 f"base63={p['base'][63]['m']:.4f}/{p['base'][63]['p']:.2f} base126={p['base'][126]['m']:.4f} vol={p['base_vol']:.3f}")
        L.append("ZONES share m21 m63 m126 p63 vol dd")
        for _, z in p["zones"].iterrows():
            L.append(f" {z.zone} {z.share:.2f} {z.m21:.4f} {z.m63:.4f} {z.m126:.4f} {z.p63:.2f} {z.vol:.3f} {z.dd:.3f}")
        L.append("SIGNALS n m21 m63 p63 pv63 m126 dd")
        for _, s in p["signals"].iterrows():
            L.append(f" {s.signal.strip()[:34]} | {s.n} {s.m21:.4f} {s.m63:.4f} {s.p63:.2f} {s.pv63:.2f} {s.m126:.4f} {s.dd:.3f}")
        L.append("GRID " + "; ".join(f"{g.rule}: n={g.n} m63={g.m63:.4f} p={g.p63:.2f}" for _, g in p["grid"].iterrows()))
        L.append("UV_EPISODES " + "; ".join(f"{d:%Y-%m-%d} {r.r63:+.3f}/{r.r126:+.3f}/dd{r.dd63:+.3f}" for d, r in p["episodes"].iterrows()))
        L.append("STRATS cagr vol sharpe maxdd invested trades")
        for _, s in p["strats"].iterrows():
            L.append(f" {s.rule} | {s.cagr:.3f} {s.vol:.3f} {s.sharpe:.2f} {s.maxdd:.3f} {s.invested:.2f} {s.trades}")
    return "\n".join(L)

# ============================================================== run
mem = globals().get("px_data")
if isinstance(mem, dict) and "QQQ" in mem and mem["Close"].index[0] <= pd.Timestamp(START_DATE) + pd.Timedelta(days=15):
    step("Using prices already in memory from the previous run (no download).")
    px_data = mem
else:
    px_data = load_data()
step("Step 2 of 4: computing the breadth indicators for every day...")
breadth, _ = add_score(compute_breadth(px_data))
qqq = px_data["QQQ"].dropna()
step(f"Step 3 of 4: breadth history {breadth.index[0]:%Y-%m-%d} to {breadth.index[-1]:%Y-%m-%d}. Testing the relabeled signals...")
df_all, events, raw_sig = build(breadth, qqq)
P = {}
if EXTEND_HISTORY and df_all.index[0] <= pd.Timestamp(HOLDOUT[0]) + pd.Timedelta(days=30):
    P["hold"] = analyze_period(df_all, events, raw_sig, *HOLDOUT)
elif EXTEND_HISTORY:
    step(f"Breadth history only starts {df_all.index[0]:%Y-%m-%d}; skipping the 2006-2016 test.")
P["dsc"] = analyze_period(df_all, events, raw_sig, *DISCOVERY)
cs = combined_signals(P)
last = df_all.index[-1]
now = {"date": last, "score": df_all.score.iloc[-1], "zone": df_all.zone.iloc[-1], "pct50": df_all.pct50.iloc[-1],
       "events_now": {k: bool(ev.iloc[-20:].any() or raw_sig[k].fillna(False).iloc[-1]) for k, ev in events.items()}}
step("Step 4 of 4: building the report...")
report_html = render(P, cs, df_all, now)
display(HTML(report_html))
charts(P, df_all, events)
with open("breadth_signals_v2_report.html", "w") as f:
    f.write(f"<html><body style='background:{BG};margin:0;padding:16px'>{report_html}</body></html>")
print("\nSaved breadth_signals_v2_report.html (Files panel on the left).")
print("\n" + "=" * 30 + " COPY EVERYTHING BELOW THIS LINE AND PASTE IT BACK TO CLAUDE " + "=" * 30)
print(paste_back(P, cs, now))
