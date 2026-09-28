# QQQ Bayesian forward-return model (3 / 6 / 9 / 12 months)

- `QQQ_Bayesian_Forecast.ipynb`: open in Google Colab and choose Run all. It asks you to upload the QQQ xlsx, then pulls QQQ back to 1999 with `yfinance` and splices it onto your file.
- `qqq_bayesian_forecast.py`: the same code as a plain script (`# %%` cells).
- `run_output.txt` / `qqq_bayes_forecast.png`: a run on the uploaded export only (2016-09-27 → 2026-09-25). It had no pre-2016 data because this sandbox had no internet.

## Method
Signals, all known on the day they are measured: price vs the 200-day average, distance from the all-time high, 12-month momentum, RSI(14), and 21-day realized volatility.

1. **Bayes tables:** `P(Up|bin) = P(bin|Up)·P(Up) / P(bin)` for each signal, using quintile bins and Laplace smoothing.
2. **Naive Bayes:** `log O(Up|x) = log O(Up) + Σ log LR_i`.
3. **Analog model with a Normal–Inverse-Gamma prior:** days with a similar chart state are weighted with a tricube kernel. A conjugate update then gives a Student-t posterior predictive for the forward log return. Overlapping windows are not independent, so the effective sample size counts horizon-length blocks instead of days.
4. **Walk-forward test:** each forecast trains only on outcomes already known at that date. Forecasts are scored with the Brier skill score against the base rate.
