"""EIGEN/USDT 15m breakout+RSI strategy, v1 (raw spec, no 1h trend filter).

All cross-timeframe values (4h ATR, 1h ADX) are attached to each 15m bar using
its CLOSE time and merge_asof(direction='backward'), so a 15m bar only ever
sees a higher-timeframe bar that had already closed by that point in time.
"""
import pandas as pd

from indicators import adx, atr, rolling_high, rolling_low, rsi


def align_lower_freq(series, bar_duration, target_close_times):
    """Reindex a lower-frequency (open-time indexed) series onto target
    close-times, using only bars whose OWN close-time <= target close-time."""
    s = series.dropna().copy()
    s.index = s.index + bar_duration
    s = s.sort_index()
    right = pd.DataFrame({"v": s.values}, index=s.index).reset_index(names="t")
    left = pd.DataFrame({"t": target_close_times})
    merged = pd.merge_asof(left, right, on="t", direction="backward")
    merged.index = target_close_times
    return merged["v"]


def build_signals(df15, df1h, df4h, cfg):
    hh = rolling_high(df15["high"], cfg["range_lookback_bars"])
    ll = rolling_low(df15["low"], cfg["range_lookback_bars"])
    rng = hh - ll
    r = rsi(df15["close"], cfg["rsi_period"])
    r_hi = rolling_high(r, cfg["rsi_lookback_bars"])
    r_lo = rolling_low(r, cfg["rsi_lookback_bars"])
    rsi_pos = (r - r_lo) / (r_hi - r_lo)

    short_price_ok = df15["close"] >= ll + cfg["range_top_pct"] * rng
    long_price_ok = df15["close"] <= ll + cfg["range_bottom_pct"] * rng
    short_rsi_ok = rsi_pos >= cfg["rsi_top_threshold"]
    long_rsi_ok = rsi_pos <= cfg["rsi_bottom_threshold"]

    close_time = df15.index + pd.Timedelta(minutes=15)
    atr4h = atr(df4h, cfg["atr_period"])
    atr4h_aligned = align_lower_freq(atr4h, pd.Timedelta(hours=4), close_time)
    atr4h_aligned.index = df15.index

    adx1h = adx(df1h, cfg["adx_period"])
    adx1h_aligned = align_lower_freq(adx1h, pd.Timedelta(hours=1), close_time)
    adx1h_aligned.index = df15.index

    out = pd.DataFrame({
        "close": df15["close"], "high": df15["high"], "low": df15["low"],
        "hh": hh, "ll": ll, "range": rng, "rsi": r,
        "long_entry": (long_price_ok & long_rsi_ok).fillna(False),
        "short_entry": (short_price_ok & short_rsi_ok).fillna(False),
        "atr4h": atr4h_aligned, "adx1h": adx1h_aligned,
    }, index=df15.index)
    return out
