"""pandas_ta_classic wrappers. pandas_ta itself won't install here (its numba
dependency doesn't yet support this VPS's Python 3.14); pandas_ta_classic is a
maintained fork with an identical API, verified against numpy 2.5 / pandas 3.0.

Every rolling reference here is shift(1)'d in the caller (strategy_v3.py), not
inside this module, so pandas_ta_classic's own window convention (whether it
includes the current bar) never matters for leak-safety.
"""
import pandas_ta_classic as pta


def rsi(close, period=14):
    return pta.rsi(close, length=period)


def atr(df, period=14):
    return pta.atr(df["high"], df["low"], df["close"], length=period)


def adx(df, period=14):
    out = pta.adx(df["high"], df["low"], df["close"], length=period)
    return out[f"ADX_{period}"]


def donchian(df, n=20):
    out = pta.donchian(df["high"], df["low"], lower_length=n, upper_length=n)
    return out[f"DCL_{n}_{n}"], out[f"DCU_{n}_{n}"]
