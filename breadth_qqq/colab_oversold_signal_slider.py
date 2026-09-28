# Oversold signal vs buy & hold -- one Colab cell with sliders.
# Signal: the dashboard's "10 Day Ratio" (stocks up 4%+ vs down 4%+ over 10 days) at or below a threshold you pick.
# Trade rule: buy QQQ at the close the day AFTER the signal (no peeking), hold it for the period you pick, then go to cash.
#             A new signal while you are already holding extends the hold. Cash earns CASH_YIELD (0% by default).
# Everything is compared with simply buying QQQ and holding it over the same dates.
# Paste into ONE Colab cell and press play. Reuses prices already in memory from the earlier cells if they are there.

!pip -q install -U yfinance

THRESHOLD = 0.5          # starting value of the threshold slider
HOLD_DAYS = 63           # starting hold period in trading days (21 = 1 month, 63 = 3 months, 126 = 6 months)
START_DATE = "2005-01-01"
START_YEAR = None        # starting value of the start-year slider (None = earliest year)
CASH_YIELD = 0.0         # yearly interest earned while in cash, e.g. 0.03 for 3%
USE_SLIDERS = True       # if the charts don't show up under the sliders, set False and change the three values above instead
FORCE_REDOWNLOAD = False

import io, os, glob, time, pickle, warnings
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import ipywidgets as widgets
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

# ---------------------------------------------------------------- the 10 Day Ratio (same formula as the dashboard), computed once
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
df["ret"] = df.qqq.pct_change().fillna(0)
df["above200"] = df.qqq > df.qqq.rolling(200).mean()
CASH_DAILY = (1 + CASH_YIELD) ** (1 / 252) - 1

# ---------------------------------------------------------------- backtest
def position(threshold, hold):
    # signal at day t's close -> buy at day t+1's close -> first return earned on day t+2, held for `hold` days
    sig = (df.ratio10 <= threshold).astype(float)
    return sig.shift(2).rolling(hold, min_periods=1).max().fillna(0)

def stats(r):
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    return dict(eq=eq, total=eq.iloc[-1] - 1, cagr=eq.iloc[-1] ** (1 / yrs) - 1, mdd=(eq / eq.cummax() - 1).min(),
                dd=eq / eq.cummax() - 1)

def backtest(threshold, hold, start_year):
    pos = position(threshold, hold)
    w = df.index >= pd.Timestamp(f"{start_year}-01-01")
    d, pos = df[w].copy(), pos[w]
    d.iloc[0, d.columns.get_loc("ret")] = 0.0
    strat = pos * d.ret + (1 - pos) * CASH_DAILY
    s, b = stats(strat), stats(d.ret)
    # trades = each unbroken stretch of days in QQQ
    p = pos.values
    starts = np.flatnonzero((p == 1) & (np.r_[0, p[:-1]] != 1))
    ends = np.flatnonzero((p == 1) & (np.r_[p[1:], 0] != 1))
    trades = []
    for k0, k1 in zip(starts, ends):
        e = max(k0 - 1, 0)                                            # bought at this close
        sd = df.index[max(df.index.get_loc(d.index[k0]) - 2, 0)]     # the signal day
        seg = d.qqq.iloc[e:k1 + 1]
        trades.append(dict(signal=sd, ratio=df.ratio10.loc[sd], above200=bool(df.above200.loc[sd]),
                           buy=d.index[e], buy_px=d.qqq.iloc[e], sell=d.index[k1], sell_px=d.qqq.iloc[k1],
                           days=k1 - e, ret=d.qqq.iloc[k1] / d.qqq.iloc[e] - 1, dip=seg.min() / seg.iloc[0] - 1,
                           open=(k1 == len(d) - 1)))
    trades = pd.DataFrame(trades)
    closed = trades[~trades.open] if len(trades) else trades
    fwd = (d.qqq.shift(-hold) / d.qqq - 1).dropna()
    out_ret = (1 + d.ret[pos == 0]).prod() - 1                        # what QQQ did while you sat in cash
    in_ret = (1 + d.ret[pos == 1]).prod() - 1
    return dict(d=d, pos=pos, s=s, b=b, trades=trades, closed=closed, typical=fwd.mean(), typical_up=(fwd > 0).mean(),
                invested=pos.mean(), out_ret=out_ret, in_ret=in_ret, days_in=int(pos.sum()))

# ---------------------------------------------------------------- display pieces
BG, INK, MUTED, GRID = "#151a21", "#e6edf3", "#8b949e", "#2a313c"
BLUE, GREEN, RED, AMBER = "#3987e5", "#2e9642", "#d0463f", "#c98500"
FONT = "Inter,'Segoe UI',Roboto,Arial,sans-serif"
HOLDS = [("1 week", 5), ("2 weeks", 10), ("1 month", 21), ("2 months", 42), ("3 months", 63),
         ("6 months", 126), ("9 months", 189), ("1 year", 252)]
pc = lambda v: "–" if pd.isna(v) else f"{v:+.1%}"
money = lambda v: f"${v:,.0f}"
col = lambda v: MUTED if pd.isna(v) else (GREEN if v > 0 else RED)

def card(title, color, rows):
    body = "".join(f'<div style="display:flex;justify-content:space-between;gap:18px;padding:3px 0">'
                   f'<span style="color:{MUTED}">{k}</span><span style="font-family:monospace;color:{c or INK}">{v}</span></div>'
                   for k, v, c in rows)
    return (f'<div style="flex:1 1 300px;background:#1c222b;border-radius:10px;padding:14px 16px;border-top:3px solid {color}">'
            f'<div style="font-weight:600;margin-bottom:6px">{title}</div>{body}</div>')

def summary_html(r, threshold, hold_label, start_year):
    s, b, cl = r["s"], r["b"], r["closed"]
    per_yr_in = (1 + r["in_ret"]) ** (252 / r["days_in"]) - 1 if r["days_in"] else np.nan
    win = (cl.ret > 0).mean() if len(cl) else np.nan
    avg = cl.ret.mean() if len(cl) else np.nan
    end_s, end_b = 10_000 * (1 + s["total"]), 10_000 * (1 + b["total"])
    better = "MORE" if end_s > end_b else "LESS"
    head = (f'Acting on every signal (10 Day Ratio ≤ {threshold:.2f}, hold {hold_label}) since {r["d"].index[0]:%b %Y} '
            f'turned $10,000 into <b style="color:{GREEN}">{money(end_s)}</b>. Buying QQQ and holding turned it into '
            f'<b style="color:{BLUE}">{money(end_b)}</b>. The signals made <b>{better}</b> money, with QQQ held only '
            f'{r["invested"]:.0%} of the time.')
    c1 = card("Acting on the signals", GREEN, [
        ("$10,000 became", money(end_s), None),
        ("Return per year", pc(s["cagr"]), col(s["cagr"])),
        ("Worst drop from a high", pc(s["mdd"]), RED),
        ("Time in QQQ", f'{r["invested"]:.0%}', None),
        ("Return per year while in QQQ", pc(per_yr_in), col(per_yr_in)),
        ("Trades (finished)", f'{len(r["trades"])} ({len(cl)})', None),
        ("Trades that made money", "–" if pd.isna(win) else f"{win:.0%}", None)])
    c2 = card("Buy QQQ and hold", BLUE, [
        ("$10,000 became", money(end_b), None),
        ("Return per year", pc(b["cagr"]), col(b["cagr"])),
        ("Worst drop from a high", pc(b["mdd"]), RED),
        ("Time in QQQ", "100%", None),
        ("QQQ while signals had you in", pc(r["in_ret"]), col(r["in_ret"])),
        ("QQQ while signals had you in cash", pc(r["out_ret"]), col(r["out_ret"])),
        ("", "", None)])
    c3 = card(f"One trade vs a random {hold_label} in QQQ", AMBER, [
        ("Average signal trade", pc(avg), col(avg)),
        (f"Average {hold_label} in QQQ (any day)", pc(r["typical"]), col(r["typical"])),
        ("Signal trades that made money", "–" if pd.isna(win) else f"{win:.0%}", None),
        (f"Any {hold_label} that made money", f'{r["typical_up"]:.0%}', None),
        ("Median worst dip inside a trade", pc(cl.dip.median()) if len(cl) else "–", RED),
        ("Best trade", pc(cl.ret.max()) if len(cl) else "–", GREEN),
        ("Worst trade", pc(cl.ret.min()) if len(cl) else "–", RED)])
    t = df.iloc[-1]
    now = (f'Today ({df.index[-1]:%b %d, %Y}) the 10 Day Ratio is <b>{t.ratio10:.2f}</b>: '
           + (f'<b style="color:{GREEN}">signal ON</b> at this threshold.' if t.ratio10 <= threshold
              else f'no signal at {threshold:.2f}.')
           + (f' The rule currently has you <b>in QQQ</b>.' if r["pos"].iloc[-1] == 1 else ' The rule currently has you <b>in cash</b>.'))
    note = (f'Not included: trading costs, taxes, and any interest on cash{"" if CASH_YIELD == 0 else f" (cash earns {CASH_YIELD:.1%}/yr here)"}. '
            'Moving the sliders until the past looks best is curve-fitting; 0.5 was the value that held up in both decades.')
    return (f'<div style="background:{BG};color:{INK};font-family:{FONT};padding:16px;border-radius:10px;font-size:14px">'
            f'<div style="font-size:16px;line-height:1.5;margin-bottom:12px">{head}</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:12px">{c1}{c2}{c3}</div>'
            f'<div style="margin-top:12px">{now}</div><div style="margin-top:6px;color:{MUTED};font-size:12px">{note}</div></div>')

def fig_growth(r):
    d, s, b = r["d"], r["s"], r["b"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
                        subplot_titles=("Growth of $10,000 (log scale)", "Drop from the previous high"))
    for name, st, c in [("Buy QQQ and hold", b, BLUE), ("Acting on the signals", s, GREEN)]:
        fig.add_trace(go.Scatter(x=d.index, y=10_000 * st["eq"], name=name, line=dict(color=c, width=2),
                                 hovertemplate=f"{name}<br>%{{x|%b %d, %Y}}: $%{{y:,.0f}}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=st["dd"], showlegend=False, line=dict(color=c, width=1.3),
                                 hovertemplate=f"{name}<br>%{{x|%b %d, %Y}}: %{{y:.1%}}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(type="log", gridcolor=GRID, tickprefix="$", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(gridcolor=GRID, tickformat=".0%", row=2, col=1)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=600, hovermode="x unified",
                      font=dict(family=FONT, color=INK, size=12), legend=dict(orientation="h", y=1.08, x=0, yanchor="bottom"),
                      margin=dict(l=70, r=30, t=70, b=40))
    fig.update_annotations(font=dict(size=12, color=MUTED))
    return fig

def fig_signals(r, threshold, hold_label):
    d, tr = r["d"], r["trades"]
    held = d.qqq.where(r["pos"] == 1)
    held = held.where(held.notna() | held.shift(-1).notna())            # start each green stretch at the buy day
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.05,
                        subplot_titles=(f"QQQ (log scale). Green = holding QQQ for {hold_label} after a signal; triangles = buy days",
                                        f"10 Day Ratio. Signal fires at or below the dashed line ({threshold:.2f})"))
    fig.add_trace(go.Scatter(x=d.index, y=d.qqq, name="QQQ (in cash)", line=dict(color=BLUE, width=1.4),
                             hovertemplate="%{x|%b %d, %Y}<br>QQQ %{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=held, name="Holding QQQ", line=dict(color=GREEN, width=2.6), connectgaps=False,
                             hoverinfo="skip"), row=1, col=1)
    for name, keep, c in [("Buy (trade made money)", tr.ret > 0, GREEN), ("Buy (trade lost money)", tr.ret <= 0, RED)] if len(tr) else []:
        t = tr[keep]
        if not len(t):
            continue
        cd = np.c_[t.signal.dt.strftime("%b %d, %Y"), t.sell.dt.strftime("%b %d, %Y"), [pc(v) for v in t.ret],
                   [pc(v) for v in t.dip], np.where(t.open, " (still open)", "")]
        fig.add_trace(go.Scatter(x=t.buy, y=t.buy_px, mode="markers", name=name, customdata=cd,
                                 marker=dict(symbol="triangle-up", size=11, color=c, line=dict(color=BG, width=1.5)),
                                 hovertemplate="<b>Bought %{x|%b %d, %Y}</b> at %{y:,.2f}<br>signal day %{customdata[0]}<br>"
                                               "sold %{customdata[1]}%{customdata[4]}<br>trade return %{customdata[2]}<br>"
                                               f"typical {hold_label} in QQQ {pc(r['typical'])}<br>worst dip inside %{{customdata[3]}}<extra></extra>"),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.ratio10, name="10 Day Ratio", line=dict(color=AMBER, width=1.1),
                             hovertemplate="%{x|%b %d, %Y}<br>10 Day Ratio %{y:.2f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=threshold, line=dict(color=GREEN, dash="dash", width=1), row=2, col=1)
    fig.update_yaxes(type="log", gridcolor=GRID, title_text="QQQ", row=1, col=1)
    fig.update_yaxes(type="log", gridcolor=GRID, title_text="ratio", tickvals=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50], row=2, col=1)
    fig.update_xaxes(gridcolor=GRID, row=2, col=1)
    fig.update_xaxes(rangeselector=dict(buttons=[dict(count=1, label="1y", step="year", stepmode="backward"),
                                                 dict(count=3, label="3y", step="year", stepmode="backward"),
                                                 dict(count=5, label="5y", step="year", stepmode="backward"),
                                                 dict(step="all", label="All")],
                                        bgcolor="#1c222b", activecolor="#2a313c", font=dict(color=INK), x=0, y=1.09), row=1, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=720, hovermode="closest",
                      font=dict(family=FONT, color=INK, size=12),
                      legend=dict(orientation="h", y=1.09, x=0.2, yanchor="bottom"), margin=dict(l=60, r=30, t=100, b=40))
    fig.update_annotations(font=dict(size=12, color=MUTED))
    return fig

def table_wrap(title, head, rows, height=460):
    return (f'<div style="background:{BG};color:{INK};font-family:{FONT};padding:14px;border-radius:10px;max-height:{height}px;'
            f'overflow:auto;margin-top:10px"><div style="font-weight:600;margin-bottom:8px">{title}</div>'
            f'<table style="border-collapse:collapse;font-size:13px"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')

def th(cols):
    return "".join(f'<th style="padding:6px 10px;text-align:{a};color:{MUTED};border-bottom:1px solid {GRID};position:sticky;top:0;'
                   f'background:{BG}">{h}</th>' for h, a in cols)

def td(v, kind="text", bold=False):
    w = "font-weight:600;" if bold else ""
    if kind == "pct":
        return f'<td style="padding:4px 10px;text-align:right;font-family:monospace;{w}color:{col(v)}">{pc(v)}</td>'
    return f'<td style="padding:4px 10px;text-align:{"left" if kind == "text" else "right"};{w}color:{INK}' \
           f'{";font-family:monospace" if kind == "num" else ""}">{v}</td>'

def sweep_html(hold, hold_label, start_year, threshold):
    rows = ""
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
        r = backtest(t, hold, start_year)
        cl, me = r["closed"], abs(t - threshold) < 1e-9
        rows += ("<tr" + (f' style="background:#223040"' if me else "") + ">"
                 + td(f"{t:.2f}" + ("  ◀ you" if me else ""), "num", me) + td(len(r["trades"]), "num", me)
                 + td(f'{r["invested"]:.0%}', "num", me) + td(money(10_000 * (1 + r["s"]["total"])), "num", me)
                 + td(r["s"]["cagr"], "pct", me) + td(r["b"]["cagr"], "pct", me) + td(r["s"]["mdd"], "pct", me)
                 + td(cl.ret.mean() if len(cl) else np.nan, "pct", me) + td(r["typical"], "pct", me) + "</tr>")
    head = th([("Threshold", "left"), ("Trades", "right"), ("Time in QQQ", "right"), ("$10k became", "right"),
               ("Signals return/yr", "right"), ("Buy & hold return/yr", "right"), ("Signals worst drop", "right"),
               ("Avg trade", "right"), (f"Avg random {hold_label}", "right")])
    return table_wrap(f"Every threshold at a {hold_label} hold since {start_year} (so you can see how touchy the result is)", head, rows, 520)

def trades_html(r, hold_label):
    tr = r["trades"]
    if not len(tr):
        return table_wrap("No signals at this threshold in this period", "", "")
    rows = "".join("<tr>" + td(f"{x.signal:%Y-%m-%d}") + td(f"{x.ratio:.2f}", "num") + td("above" if x.above200 else "below")
                   + td(f"{x.buy:%Y-%m-%d}") + td(f"{x.buy_px:,.2f}", "num") + td(f"{x.sell:%Y-%m-%d}" + (" (open)" if x.open else ""))
                   + td(f"{x.sell_px:,.2f}", "num") + td(x.days, "num") + td(x.ret, "pct", True)
                   + td(x.ret - r["typical"], "pct") + td(x.dip, "pct") + "</tr>"
                   for x in tr.iloc[::-1].itertuples())
    head = th([("Signal day", "left"), ("10 Day Ratio", "right"), ("QQQ vs 200-day", "left"), ("Bought", "left"), ("at", "right"),
               ("Sold", "left"), ("at", "right"), ("Days held", "right"), ("Trade return", "right"),
               (f"vs random {hold_label}", "right"), ("Worst dip inside", "right")])
    return table_wrap(f"Every trade, newest first (a signal while already holding extends the trade). "
                      f"Random {hold_label} in QQQ averaged {pc(r['typical'])}", head, rows)

# ---------------------------------------------------------------- sliders
first_year, last_year = df.index[0].year, df.index[-1].year
start0 = min(max(START_YEAR or first_year, first_year), last_year - 1)
hold0 = HOLD_DAYS if HOLD_DAYS in dict(HOLDS).values() else 63

def render(threshold, hold, start_year):
    label = dict((v, k) for k, v in HOLDS)[hold]
    r = backtest(threshold, hold, start_year)
    display(HTML(summary_html(r, threshold, label, start_year)))
    fig_growth(r).show()
    fig_signals(r, threshold, label).show()
    display(HTML(trades_html(r, label)))
    display(HTML(sweep_html(hold, label, start_year, threshold)))

if USE_SLIDERS:
    style, lay = {"description_width": "150px"}, widgets.Layout(width="560px")
    w_thr = widgets.FloatSlider(value=THRESHOLD, min=0.2, max=1.5, step=0.05, readout_format=".2f", continuous_update=False,
                                description="Signal threshold ≤", style=style, layout=lay)
    w_hold = widgets.SelectionSlider(options=HOLDS, value=hold0, continuous_update=False, description="Hold QQQ for",
                                     style=style, layout=lay)
    w_start = widgets.IntSlider(value=start0, min=first_year, max=last_year - 1, continuous_update=False,
                                description="Start year", style=style, layout=lay)
    out = widgets.interactive_output(render, {"threshold": w_thr, "hold": w_hold, "start_year": w_start})
    display(widgets.VBox([widgets.HTML("<b>Move a slider and let go. Everything below redraws in a few seconds.</b>"),
                          w_thr, w_hold, w_start]), out)
else:
    render(THRESHOLD, hold0, start0)
