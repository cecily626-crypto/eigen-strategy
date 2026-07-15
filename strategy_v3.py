"""EIGEN/USDT v3: intraday mean-reversion rebuild.

Entry needs ALL THREE gates true on the same 15m bar:
  1. deep extreme:  short price >= LL+0.95*Range, long price <= LL+0.05*Range
  2. RSI hard filter: short (RSI top 10% of its own 20-bar range) OR RSI>=75
                       long  (RSI bottom 10% of its own 20-bar range) OR RSI<=25
  3. range confirmation: 1h ADX < adx_max_for_entry (skip trend segments entirely)

Cooldown state (universal 4-bar, +8-bar same-direction after a stop-loss) is
sequential/history-dependent, so it lives in backtest_v3.py's event loop, not
here. This module only produces the raw per-bar candidate signals.
"""
import pandas as pd

from indicators_ta import adx, atr, donchian, rsi
from strategy import align_lower_freq
from indicators import rolling_high, rolling_low


def build_signals_v3(df15, df1h, cfg):
    dcl, dcu = donchian(df15, cfg["range_lookback_bars"])
    ll, hh = dcl.shift(1), dcu.shift(1)
    rng = hh - ll

    r = rsi(df15["close"], cfg["rsi_period"])
    r_hi = rolling_high(r, cfg["rsi_lookback_bars"])
    r_lo = rolling_low(r, cfg["rsi_lookback_bars"])
    rsi_pos = (r - r_lo) / (r_hi - r_lo)

    deep_short = df15["close"] >= ll + cfg["range_top_pct"] * rng
    deep_long = df15["close"] <= ll + cfg["range_bottom_pct"] * rng
    rsi_short_ok = (rsi_pos >= cfg["rsi_top_pos_threshold"]) | (r >= cfg["rsi_abs_overbought"])
    rsi_long_ok = (rsi_pos <= cfg["rsi_bottom_pos_threshold"]) | (r <= cfg["rsi_abs_oversold"])

    close_time = df15.index + pd.Timedelta(minutes=15)
    adx1h = adx(df1h, cfg["adx_period"])
    adx1h_aligned = align_lower_freq(adx1h, pd.Timedelta(hours=1), close_time)
    adx1h_aligned.index = df15.index
    regime_ok = adx1h_aligned < cfg["adx_max_for_entry"]

    atr15 = atr(df15, cfg["atr_period"])

    long_entry = (deep_long & rsi_long_ok & regime_ok).fillna(False)
    short_entry = (deep_short & rsi_short_ok & regime_ok).fillna(False)

    return pd.DataFrame({
        "close": df15["close"], "high": df15["high"], "low": df15["low"],
        "hh": hh, "ll": ll, "range": rng, "rsi": r,
        "long_entry": long_entry, "short_entry": short_entry,
        "atr15": atr15, "adx1h": adx1h_aligned,
    }, index=df15.index)
