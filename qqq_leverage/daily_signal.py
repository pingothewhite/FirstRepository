# ============================================================
#  QQQ DAILY SIGNAL — what leverage to hold tomorrow
#  Run after the close (Colab or local). Same rules as the v6
#  dashboard's recommended mode:
#    regime ON  -> leverage from volatility buckets (default):
#                    vol < 18% -> 1.6x | 18-22% -> 1.4x | 22-26% -> 1.2x | >= 26% -> 1.0x
#                  (or continuous: clip(vol_target / vol, 1.0, 1.6))
#                  +0.4 while QQQ <= 80% of its 52-week high
#                  +0.3 more while QQQ <= 75% of it
#                  trade only when the target moves >= 0.1
#    regime OFF -> 100% SGOV
#  Regime: 200-day SMA band; ON after 3 closes above SMA x 1.04,
#  OFF after 15 closes below SMA x 0.92 (counts reset as in v5).
#  WHEN TO RUN: every Friday after the close, PLUS any day QQQ closes
#  up or down 3% or more, PLUS daily once QQQ closes below the
#  regime exit line (SMA x 0.92). Tested: this matches daily checking.
#  NOT FINANCIAL ADVICE.
# ============================================================
current_leverage = 1.3     # what you hold right now (0 if in SGOV)
use_buckets = True
buckets = [(0.18, 1.6), (0.22, 1.4), (0.26, 1.2), (9.99, 1.0)]
vol_target, lev_min, lev_max = 0.28, 1.0, 1.6
dip1, add1, dip2, add2 = 0.20, 0.4, 0.25, 0.3
rebalance_band = 0.10

import glob, os, subprocess, sys
import numpy as np
import pandas as pd


def load():
    f = os.environ.get("QQQ_FILE", "")
    if not f:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yfinance"], check=False)
            import yfinance as yf
            q = yf.download("QQQ", start="1999-03-10", auto_adjust=False, progress=False)
            if isinstance(q.columns, pd.MultiIndex):
                q.columns = q.columns.get_level_values(0)
            if len(q) > 1000:
                return q["Close"].dropna()
        except Exception:
            pass
        f = (sorted(glob.glob("*QQQ*.xlsx") + glob.glob("*QQQ*.csv")) or [None])[0]
    d = pd.read_excel(f) if f.endswith("xlsx") else pd.read_csv(f)
    return d.assign(Date=pd.to_datetime(d["Date"])).set_index("Date")["Close"].sort_index()


c = load()
p = c.values.astype(float)
sma = pd.Series(p).rolling(200).mean().values
upper, lower = sma * 1.04, sma * 0.92
state = 1 if p[199] > sma[199] else 0
a = b = 0
for i in range(199, len(p)):
    if p[i] > upper[i]:   a += 1; b = 0
    elif p[i] < lower[i]: b += 1; a = 0
    if state == 0 and a >= 3:    state = 1; a = b = 0
    elif state == 1 and b >= 15: state = 0; a = b = 0

r = np.r_[0.0, p[1:] / p[:-1] - 1]
lam = 0.5 ** (1 / 10)
v = np.mean(r[1:22] ** 2)
for x in r:
    v = lam * v + (1 - lam) * x * x
vol = np.sqrt(252 * v)
hi52 = p[-252:].max()
last = p[-1]

target = 0.0
if state == 1:
    target = next(l for c, l in buckets if vol < c) if use_buckets else float(np.clip(vol_target / vol, lev_min, lev_max))
    target += add1 if last <= hi52 * (1 - dip1) else 0.0
    target += add2 if last <= hi52 * (1 - dip2) else 0.0
    target = min(target, 2.0)
trade = (target == 0) != (current_leverage == 0) or abs(target - current_leverage) >= rebalance_band - 1e-9
new_L = target if trade else current_leverage

print(f"QQQ {last:,.2f} on {c.index[-1].date()} | 200-SMA {sma[-1]:,.2f} ({last/sma[-1]-1:+.1%})")
print(f"Regime: {'ON' if state else 'OFF (SGOV)'} | closes above band {a}/3, below band {b}/15")
print(f"Volatility (10-day half-life): {vol:.1%} | 52-week high {hi52:,.2f} ({last/hi52-1:+.1%})")
print(f"Target leverage {target:.2f}x -> {'TRADE to ' + format(new_L, '.2f') + 'x' if trade else 'HOLD ' + format(current_leverage, '.2f') + 'x (inside 0.1 band)'}")
if new_L >= 1:
    print(f"  Mix: {new_L-1:.0%} QLD + {2-new_L:.0%} QQQ" if new_L <= 2 else "  Mix: 100% QLD")
print("\nPrice levels that change the rules (SMA drifts slowly; re-run daily):")
print(f"  Regime exit zone : closes below {lower[-1]:,.2f} (15 of them -> SGOV)")
print(f"  Regime re-entry  : 3 closes above {upper[-1]:,.2f} when OFF")
print(f"  Dip add +{add1}x    : QQQ at or below {hi52*(1-dip1):,.2f}")
print(f"  Dip add +{add2}x more: QQQ at or below {hi52*(1-dip2):,.2f}")
print("\nVolatility buckets: <18% 1.6x | 18-22% 1.4x | 22-26% 1.2x | >=26% 1.0x  (+0.4 at -20%, +0.3 more at -25%)")
print(f"Extra check trigger: any close beyond {last*0.97:,.2f} / {last*1.03:,.2f} (3% move from today)")
