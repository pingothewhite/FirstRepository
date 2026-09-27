"""Download daily bars for the S&P 1500 and build the breadth history used by analyze.py.

    python build_history.py            # writes data/prices.pkl and data/breadth_history.csv
"""
import io
import os
import pickle
import sys
import time

import pandas as pd
import requests
import yfinance as yf

from breadth_core import add_score, compute_breadth

START = "2014-06-01"          # a year+ of warm-up before the Sept 2016 start of the QQQ file
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


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


def download_prices(tickers, chunk=100, pause=1.0):
    frames = []
    for i in range(0, len(tickers), chunk):
        batch, d = tickers[i:i + chunk], None
        for attempt in range(4):
            try:
                d = yf.download(batch, start=START, interval="1d", auto_adjust=True,
                                group_by="column", threads=True, progress=False)
                break
            except Exception as e:
                print(f"\n  batch {i // chunk + 1}: retry {attempt + 1} ({e})")
                time.sleep(5 * (attempt + 1))
        if d is not None and len(d):
            if not isinstance(d.columns, pd.MultiIndex):
                d.columns = pd.MultiIndex.from_product([d.columns, batch])
            frames.append(_clean_index(d))
        print(f"  downloaded {min(i + chunk, len(tickers)):,}/{len(tickers):,}", end="\r", flush=True)
        time.sleep(pause)
    raw = pd.concat(frames, axis=1)
    return raw.loc[:, ~raw.columns.duplicated()]


def main():
    os.makedirs(DATA, exist_ok=True)
    cache = os.path.join(DATA, "prices.pkl")
    if os.path.exists(cache) and "--refresh" not in sys.argv:
        with open(cache, "rb") as f:
            px = pickle.load(f)
        print("Using cached prices.")
    else:
        tickers = sp1500()
        print(f"Universe: {len(tickers):,} tickers (current S&P 1500 members)")
        raw = download_prices(tickers)
        px = {f: raw[f] for f in ["Close", "High", "Low", "Volume"]}
        good = px["Close"].notna().sum() >= 60
        px = {f: v.loc[:, good[good].index] for f, v in px.items()}
        spx = _clean_index(yf.download("^GSPC", start=START, auto_adjust=True, progress=False))["Close"]
        px["SPX"] = spx.iloc[:, 0] if isinstance(spx, pd.DataFrame) else spx
        with open(cache, "wb") as f:
            pickle.dump(px, f)
        print(f"\nGot usable data for {good.sum():,} tickers.")

    b, comp = add_score(compute_breadth(px))
    comp.columns = ["c_" + c for c in comp.columns]
    out = b.join(comp)
    out.index.name = "Date"
    out.to_csv(os.path.join(DATA, "breadth_history.csv"))
    print(f"Wrote {len(out):,} days ({out.index[0]:%Y-%m-%d} to {out.index[-1]:%Y-%m-%d}).")


if __name__ == "__main__":
    main()
