"""Same walk-forward truncation-consistency proof as validate_no_lookahead.py,
adapted to the v3 signal set (build_signals_v3 / backtest_v3.load_full)."""
import os
import sys

import numpy as np
import pandas as pd
import yaml

from backtest_v3 import load_full
from strategy_v3 import build_signals_v3

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["hh", "ll", "range", "rsi", "long_entry", "short_entry", "atr15", "adx1h"]


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config_v3.yaml"
    with open(os.path.join(HERE, config_path)) as f:
        cfg = yaml.safe_load(f)
    df15, df1h = load_full(cfg)
    full_sig = build_signals_v3(df15, df1h, cfg)

    cutoff_positions = [int(len(df15) * p) for p in (0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 0.97)]
    results = []
    for pos in cutoff_positions:
        t_cut = df15.index[pos]
        d15 = df15[df15.index <= t_cut]
        d1h = df1h[df1h.index <= t_cut]
        trunc_sig = build_signals_v3(d15, d1h, cfg)

        row_full = full_sig.loc[t_cut, COLS]
        row_trunc = trunc_sig.loc[t_cut, COLS]
        ok, diffs = True, {}
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
    print("V3 NO-LOOKAHEAD VALIDATION:", "PASS" if all_pass else "FAIL")
    for r in results:
        print(f"  {r['cutoff']}: {'OK' if r['pass'] else 'MISMATCH ' + str(r['diffs'])}")
    return all_pass


if __name__ == "__main__":
    main()
