"""MFE (max favorable excursion) diagnosis for v3 original, edge-eval OOS trades.

For each trade, walks the 15m bars from just after entry through the exit bar
and finds the best price reached in the FAVORABLE direction. MFE is expressed
as a fraction of the entry->midline (tp1) distance, so 1.0 means price reached
exactly the midline at some point during the trade (whether or not it was
still there at exit), and >1.0 means it went past the midline toward tp2.
"""
import os

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def compute_mfe(trades_path, symbol):
    df15 = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}_15m_full.csv"), index_col=0, parse_dates=True)
    trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"])

    mfe_pcts = []
    for _, tr in trades.iterrows():
        entry_pos = df15.index.get_loc(tr["entry_time"])
        exit_open_time = tr["exit_time"] - pd.Timedelta(minutes=15)
        exit_pos = df15.index.get_loc(exit_open_time)
        path = df15.iloc[entry_pos + 1: exit_pos + 1]

        entry_price, tp1_price = tr["entry_price"], tr["tp1_price"]
        target_distance = abs(tp1_price - entry_price)
        if len(path) == 0 or target_distance == 0:
            mfe_pcts.append(0.0)
            continue
        if tr["side"] == "long":
            favorable_move = path["high"].max() - entry_price
        else:
            favorable_move = entry_price - path["low"].min()
        mfe_pcts.append(favorable_move / target_distance)

    trades = trades.copy()
    trades["mfe_pct"] = mfe_pcts
    return trades


def bucket_report(trades, label):
    n = len(trades)
    print(f"--- {label} (n={n}) ---")
    for th in (0.25, 0.50, 0.75, 1.00):
        frac = (trades["mfe_pct"] >= th).mean()
        print(f"  reached >= {int(th*100)}% of entry->midline distance: {frac*100:.1f}%")
    print(f"  mean mfe_pct={trades['mfe_pct'].mean():.3f}  median={trades['mfe_pct'].median():.3f}")


def main():
    with open(os.path.join(HERE, "config_v3.yaml")) as f:
        cfg = yaml.safe_load(f)
    trades_path = os.path.join(HERE, "trades_v3_edge_eval_oos.csv")
    trades = compute_mfe(trades_path, cfg["symbol"])
    trades.to_csv(os.path.join(HERE, "trades_v3_oos_with_mfe.csv"), index=False)

    bucket_report(trades, "ALL OOS trades")
    losers = trades[trades["net_pnl"] <= 0]
    winners = trades[trades["net_pnl"] > 0]
    bucket_report(losers, "LOSING OOS trades only")
    bucket_report(winners, "WINNING OOS trades only")


if __name__ == "__main__":
    main()
