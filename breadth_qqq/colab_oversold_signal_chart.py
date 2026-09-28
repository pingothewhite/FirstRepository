# Oversold signal vs QQQ -- one simple Colab cell.
# Signal: the dashboard's "10 Day Ratio" (stocks up 4%+ vs down 4%+ over 10 days) drops to 0.5 or lower.
# This was the one breadth signal that held up in both 2006-2016 and 2016-2026.
# Paste into ONE Colab cell and press play. Reuses prices already in memory from the earlier cells if they are there.

!pip -q install -U yfinance

THRESHOLD = 0.5          # signal fires when the 10 Day Ratio is at or below this
GAP = 20                 # ignore repeat signals within this many trading days (one marker per episode)
START_DATE = "2005-01-01"
FORCE_REDOWNLOAD = False

import io, os, glob, time, pickle, warnings
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import HTML, display
warnings.filterwarnings("ignore")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

# ---------------------------------------------------------------- data (reuse if possible)
def _clean(df):
    df = df.copy()
    idx = pd.to_datetime(df.index)
    df.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()

def _sp1500():
    out = []
    for page in ["List_of_S%26P_500_companies", "List_of_S%26P_400_companies", "List_of_S%26P_600_companies"]:
        html = requests.get(f"https://en.wikipedia.org/wiki/{page}", headers=UA, timeout=30).text
        for t in pd.read_html(io.StringIO(html)):
            col = next((c for c in t.columns if "symbol" in str(c).lower() or "ticker" in str(c).lower()), None)
            if col is not None:
                s = t[col].astype(str).str.strip()
                s = s[s.str.fullmatch(r"[A-Z][A-Z.\-]{0,6}")]
                if len(s) > 50:
                    out += s.tolist()
                    break
    return sorted({s.replace(".", "-") for s in out})

def _download():
    tickers = _sp1500()
    print(f"Downloading daily bars since {START_DATE} for {len(tickers):,} stocks (about 5-15 minutes)...")
    frames = []
    for i in range(0, len(tickers), 100):
        d = yf.download(tickers[i:i + 100], start=START_DATE, auto_adjust=True, group_by="column", threads=True, progress=False)
        frames.append(_clean(d))
        print(f"  {min(i + 100, len(tickers)):,} of {len(tickers):,}", flush=True)
        time.sleep(1)
    raw = pd.concat(frames, axis=1)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    q = _clean(yf.download("QQQ", start=START_DATE, auto_adjust=True, progress=False))["Close"]
    px = {"Close": raw["Close"], "Volume": raw["Volume"], "QQQ": q.iloc[:, 0] if isinstance(q, pd.DataFrame) else q}
    with open("oversold_signal_prices.pkl", "wb") as f:
        pickle.dump({"px": px}, f)
    return px

def load():
    mem = globals().get("px_data")
    if not FORCE_REDOWNLOAD and isinstance(mem, dict) and {"Close", "Volume", "QQQ"} <= set(mem):
        print("Using prices already in memory from the earlier cell.")
        return mem
    caches = [] if FORCE_REDOWNLOAD else sorted(glob.glob("breadth_prices_*.pkl") + glob.glob("oversold_signal_prices.pkl"))
    for path in caches:
        with open(path, "rb") as f:
            px = pickle.load(f)["px"]
        if {"Close", "Volume", "QQQ"} <= set(px):
            print(f"Using saved prices from {path} (last day {px['Close'].index[-1]:%Y-%m-%d}; "
                  f"set FORCE_REDOWNLOAD = True for fresh data).")
            return px
    return _download()

px = load()

# ---------------------------------------------------------------- the signal (same formula as the dashboard)
C, V = px["Close"], px["Volume"].reindex_like(px["Close"])
prev = C.shift(1)
pct = C / prev - 1
dv20 = (C * V).rolling(20, min_periods=15).mean()
elig = C.notna() & prev.notna() & (C >= 5) & (dv20 >= 2e6)
volok = (V > V.shift(1)) & (V >= 100_000)
up4 = ((pct >= 0.04) & volok & elig).sum(axis=1)
dn4 = ((pct <= -0.04) & volok & elig).sum(axis=1)
universe = elig.sum(axis=1)
ok = universe >= 0.5 * universe.rolling(60, min_periods=1).median()        # skip half-reported days
ratio10 = (up4[ok].rolling(10).sum() / dn4[ok].rolling(10).sum().clip(lower=1))

df = pd.DataFrame({"ratio10": ratio10}).join(px["QQQ"].rename("qqq"), how="inner").dropna()
df = df[df.index >= pd.Timestamp(START_DATE) + pd.DateOffset(months=2)]
for h, name in [(21, "m1"), (63, "m3"), (126, "m6")]:
    df[name] = df.qqq.shift(-h) / df.qqq - 1
df["dip"] = df.qqq[::-1].rolling(63, min_periods=1).min()[::-1].shift(-1) / df.qqq - 1
df["above200"] = df.qqq > df.qqq.rolling(200).mean()

raw_sig = df.ratio10 <= THRESHOLD
sig_idx, last = [], -10**9
for i, v in enumerate(raw_sig.values):
    if v and i - last > GAP:
        sig_idx.append(i); last = i
sig = df.iloc[sig_idx].copy()

# ---------------------------------------------------------------- summary numbers
done = sig.dropna(subset=["m3"])
normal3 = df.m3.mean()
print(f"\n{len(sig)} Oversold signals since {df.index[0]:%b %Y}. After 3 months QQQ was higher {(done.m3 > 0).mean():.0%} of the time, "
      f"averaging {done.m3.mean():+.1%} vs {normal3:+.1%} for any 3 months. Typical dip first: {done.dip.median():+.1%}.")
for label, part in [("QQQ above its 200-day average", done[done.above200]), ("QQQ below its 200-day average", done[~done.above200])]:
    if len(part):
        print(f"  When {label}: {len(part)} signals, {(part.m3 > 0).mean():.0%} up after 3 months, average {part.m3.mean():+.1%}.")
t = df.iloc[-1]
print(f"\nToday ({df.index[-1]:%b %d, %Y}): 10 Day Ratio = {t.ratio10:.2f} "
      f"({'SIGNAL ON' if t.ratio10 <= THRESHOLD else f'signal fires at {THRESHOLD}'}). "
      f"QQQ is {'above' if t.above200 else 'below'} its 200-day average.")

# ---------------------------------------------------------------- chart 1: QQQ with the signal, and the ratio underneath
BG, INK, MUTED, GRID = "#151a21", "#e6edf3", "#8b949e", "#2a313c"
BLUE, GREEN, RED, AMBER = "#3987e5", "#2e9642", "#d0463f", "#c98500"
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.05,
                    subplot_titles=("QQQ price (log scale). Triangles = Oversold signal; shading = the 3 months after (green up, red down)",
                                    "10 Day Ratio (stocks up 4%+ ÷ down 4%+ over 10 days). Signal fires at or below the dashed line"))
fig.add_trace(go.Scatter(x=df.index, y=df.qqq, name="QQQ", line=dict(color=BLUE, width=1.6),
                         hovertemplate="%{x|%b %d, %Y}<br>QQQ %{y:,.2f}<extra></extra>"), row=1, col=1)
for d, r in sig.iterrows():
    end = df.index[min(df.index.get_loc(d) + 63, len(df) - 1)]
    col = GREEN if (pd.notna(r.m3) and r.m3 > 0) else RED if pd.notna(r.m3) else MUTED
    fig.add_vrect(x0=d, x1=end, fillcolor=col, opacity=0.13, line_width=0, row=1, col=1)
fmt = lambda v: "n/a yet" if pd.isna(v) else f"{v:+.1%}"
fig.add_trace(go.Scatter(x=sig.index, y=sig.qqq, mode="markers", name="Oversold signal",
                         marker=dict(symbol="triangle-up", size=11, color=GREEN, line=dict(color=BG, width=1.5)),
                         customdata=np.c_[[fmt(v) for v in sig.m1], [fmt(v) for v in sig.m3], [fmt(v) for v in sig.m6], [fmt(v) for v in sig.dip]],
                         hovertemplate="<b>Signal %{x|%b %d, %Y}</b><br>QQQ %{y:,.2f}<br>next 1 mo: %{customdata[0]}<br>"
                                       "next 3 mo: %{customdata[1]}<br>next 6 mo: %{customdata[2]}<br>worst dip in 3 mo: %{customdata[3]}<extra></extra>"),
              row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.ratio10, name="10 Day Ratio", line=dict(color=AMBER, width=1.2),
                         hovertemplate="%{x|%b %d, %Y}<br>10 Day Ratio %{y:.2f}<extra></extra>"), row=2, col=1)
fig.add_trace(go.Scatter(x=sig.index, y=sig.ratio10, mode="markers", showlegend=False,
                         marker=dict(symbol="triangle-up", size=9, color=GREEN, line=dict(color=BG, width=1.5)), hoverinfo="skip"), row=2, col=1)
fig.add_hline(y=THRESHOLD, line=dict(color=GREEN, dash="dash", width=1), row=2, col=1)
fig.update_yaxes(type="log", gridcolor=GRID, title_text="QQQ", row=1, col=1)
fig.update_yaxes(type="log", gridcolor=GRID, title_text="ratio", row=2, col=1)
fig.update_xaxes(gridcolor=GRID, row=2, col=1)
fig.update_xaxes(rangeselector=dict(buttons=[dict(count=1, label="1y", step="year", stepmode="backward"),
                                             dict(count=3, label="3y", step="year", stepmode="backward"),
                                             dict(count=5, label="5y", step="year", stepmode="backward"),
                                             dict(step="all", label="All")],
                                    bgcolor="#1c222b", activecolor="#2a313c", font=dict(color=INK), x=0, y=1.09), row=1, col=1)
fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=780, hovermode="closest",
                  font=dict(family="Inter,'Segoe UI',Roboto,Arial,sans-serif", color=INK, size=12),
                  legend=dict(orientation="h", y=1.09, x=0.22, yanchor="bottom"), margin=dict(l=60, r=30, t=100, b=40))
fig.update_annotations(font=dict(size=12, color=MUTED))
fig.show()

# ---------------------------------------------------------------- chart 2: what QQQ did before and after every signal
pre, post = 20, 126
paths, trend = [], []
for d in sig.index:
    i = df.index.get_loc(d)
    if i - pre >= 0 and i + post < len(df):
        seg = df.qqq.iloc[i - pre:i + post + 1].values / df.qqq.iloc[i] * 100
        paths.append(seg); trend.append(bool(df.above200.iloc[i]))
x = np.arange(-pre, post + 1)
fig2 = go.Figure()
for p, d in zip(paths, [d for d in sig.index if df.index.get_loc(d) - pre >= 0 and df.index.get_loc(d) + post < len(df)]):
    fig2.add_trace(go.Scatter(x=x, y=p, mode="lines", line=dict(color="rgba(139,148,158,0.22)", width=1), showlegend=False,
                              hovertemplate=f"signal {d:%b %d, %Y}<br>day %{{x}}: %{{y:.1f}}<extra></extra>"))
paths, trend = np.array(paths), np.array(trend)
typical = np.array([100] * (pre + 1) + [100 * (1 + (df.qqq.shift(-k) / df.qqq - 1).mean()) for k in range(1, post + 1)])
if trend.any():
    fig2.add_trace(go.Scatter(x=x, y=paths[trend].mean(axis=0), name=f"Average when QQQ above 200-day ({trend.sum()})",
                              line=dict(color=BLUE, width=2.5)))
if (~trend).any():
    fig2.add_trace(go.Scatter(x=x, y=paths[~trend].mean(axis=0), name=f"Average when QQQ below 200-day ({(~trend).sum()})",
                              line=dict(color=RED, width=2.5)))
fig2.add_trace(go.Scatter(x=x, y=paths.mean(axis=0), name=f"Average of all {len(paths)} signals", line=dict(color=GREEN, width=3.5)))
fig2.add_trace(go.Scatter(x=x[pre:], y=typical[pre:], name="Any random day (normal drift)", line=dict(color=INK, dash="dot", width=2)))
fig2.add_vline(x=0, line=dict(color=INK, width=1), annotation_text="signal day", annotation_position="top right",
               annotation_font_color=INK)
fig2.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=520, hovermode="closest",
                   title="QQQ from 1 month before to 6 months after each signal (signal day = 100). Gray = each signal",
                   xaxis=dict(title="trading days from the signal (21 ≈ 1 month, 63 ≈ 3 months, 126 ≈ 6 months)", gridcolor=GRID),
                   yaxis=dict(title="QQQ, signal day = 100", gridcolor=GRID),
                   font=dict(family="Inter,'Segoe UI',Roboto,Arial,sans-serif", color=INK, size=12),
                   legend=dict(orientation="h", y=-0.2, x=0), margin=dict(l=60, r=30, t=60, b=40))
fig2.show()

# ---------------------------------------------------------------- table: every signal
def cell(v):
    if pd.isna(v):
        return '<td style="padding:4px 10px;text-align:right;color:#8b949e">–</td>'
    c = GREEN if v > 0 else RED
    return f'<td style="padding:4px 10px;text-align:right;color:{c};font-family:monospace">{v:+.1%}</td>'
td = f"padding:4px 10px;color:{INK}"
rows = "".join(f'<tr><td style="{td}">{d:%Y-%m-%d}</td><td style="{td};text-align:right;font-family:monospace">{r.qqq:,.2f}</td>'
               f'<td style="{td};text-align:right;font-family:monospace">{r.ratio10:.2f}</td>'
               f'<td style="{td}">{"above" if r.above200 else "below"}</td>{cell(r.m1)}{cell(r.m3)}{cell(r.m6)}{cell(r.dip)}</tr>'
               for d, r in sig.iloc[::-1].iterrows())
th = "".join(f'<th style="padding:6px 10px;text-align:{a};color:#8b949e;border-bottom:1px solid #2a313c">{h}</th>'
             for h, a in [("Signal date", "left"), ("QQQ", "right"), ("10 Day Ratio", "right"), ("QQQ vs 200-day", "left"),
                          ("Next 1 mo", "right"), ("Next 3 mo", "right"), ("Next 6 mo", "right"), ("Worst dip (3 mo)", "right")])
display(HTML(f'<div style="background:{BG};color:{INK};font-family:Inter,Segoe UI,Arial,sans-serif;padding:14px;border-radius:10px;'
             f'max-height:520px;overflow:auto"><div style="font-weight:600;margin-bottom:8px">Every Oversold signal (newest first)</div>'
             f'<table style="border-collapse:collapse;font-size:13px"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>'))
