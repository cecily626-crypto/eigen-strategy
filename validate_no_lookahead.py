"""Proves the signal pipeline has no look-ahead bias.

At several cutoff points, truncate all raw OHLCV data to "what would have been
known" at that moment and recompute every indicator from scratch. If the
signal at the cutoff bar matches the one computed from the FULL dataset, then
nothing after that point could have influenced it -> no future function.
"""
import os

import numpy as np
import pandas as pd
import yaml

from backtest import load
from strategy import build_signals

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["hh", "ll", "range", "rsi", "long_entry", "short_entry", "atr4h", "adx1h"]


def main():
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    df15, df1h, df4h = load(cfg)
    full_sig = build_signals(df15, df1h, df4h, cfg)

    cutoff_positions = [int(len(df15) * p) for p in (0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 0.97)]
    results = []
    for pos in cutoff_positions:
        t_cut = df15.index[pos]
        d15 = df15[df15.index <= t_cut]
        d1h = df1h[df1h.index <= t_cut]
        d4h = df4h[df4h.index <= t_cut]
        trunc_sig = build_signals(d15, d1h, d4h, cfg)

        row_full = full_sig.loc[t_cut, COLS]
        row_trunc = trunc_sig.loc[t_cut, COLS]
        ok = True
        diffs = {}
        for c in COLS:
            a, b = row_full[c], row_trunc[c]
            if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
                same = bool(a) == bool(b)
            elif pd.isna(a) and pd.isna(b):
                same = True
            else:
                same = abs(float(a) - float(b)) < 1e-8
            if not same:
                ok = False
                diffs[c] = (a, b)
        results.append({"cutoff": str(t_cut), "pass": ok, "diffs": diffs})

    all_pass = all(r["pass"] for r in results)
    print("NO-LOOKAHEAD VALIDATION:", "PASS" if all_pass else "FAIL")
    for r in results:
        print(f"  {r['cutoff']}: {'OK' if r['pass'] else 'MISMATCH ' + str(r['diffs'])}")
    return all_pass


if __name__ == "__main__":
    main()
