"""Event-driven 15m backtest for EIGEN/USDT v3 (mean-reversion rebuild).

Differences from v1/v2 that matter for this loop:
  - fixed stop set ONCE at entry (short: HH+0.5*ATR15m, long: LL-0.5*ATR15m),
    no trailing -- HH/LL/ATR are snapshotted at the entry bar.
  - two-stage take-profit: 50% of the position closes at the entry-time range
    midline, the remainder closes at the entry-time OPPOSITE edge of the range.
  - position sizing is risk-based: qty = (risk_pct * equity) / stop_distance,
    capped so notional never exceeds equity * leverage_cap.
  - cooldown: no new entry within `cooldown_bars_after_any_close` bars of any
    FULL close (stop-loss or the tp2 close), and an extra same-direction-only
    cooldown of `cooldown_bars_after_stop_same_dir` bars after a stop-loss.
    A tp1 partial scale-out alone does not start either cooldown -- the
    position is still open, not "closed".
  - a running peak-to-trough drawdown >= max_drawdown_pause_pct permanently
    stops new entries for the rest of the segment (no auto-resume, since the
    spec didn't define one).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

from backtest import compute_metrics, equity_stats
from strategy_v3 import build_signals_v3

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def load_full(cfg):
    symbol = cfg["symbol"]
    df15 = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}_15m_full.csv"), index_col=0, parse_dates=True)
    df1h = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}_1h_full.csv"), index_col=0, parse_dates=True)
    return df15, df1h


def run_segment(sig, cfg, start_time, end_time, dd_pause_enabled=True):
    fee = cfg["fee_pct"]
    slip = cfg["slippage_pct"]
    lev_cap = cfg["leverage_cap"]
    risk_pct = cfg["risk_pct_per_trade"]
    funding_rate = cfg["funding_rate_per_interval"]
    funding_hours = cfg["funding_interval_hours"]
    cd_any = cfg["cooldown_bars_after_any_close"]
    cd_stop_dir = cfg["cooldown_bars_after_stop_same_dir"]
    dd_pause = cfg["max_drawdown_pause_pct"]

    seg = sig[(sig.index >= start_time) & (sig.index <= end_time)]
    equity = float(cfg["account_start_usdt"])
    peak_equity = equity
    paused = False
    paused_at = None

    pos = None
    last_full_close_i = -10**9
    last_stop_i_by_side = {1: -10**9, -1: -10**9}
    trades = []
    equity_curve = []

    idx = seg.index
    for i in range(len(idx)):
        row = seg.iloc[i]
        t = idx[i]
        close_t = t + pd.Timedelta(minutes=15)
        close, high, low = row["close"], row["high"], row["low"]

        if pos is None:
            mtm = equity
        else:
            side = pos["side"]
            mtm = equity + side * (close - pos["entry_price"]) * pos["qty_remaining"]

            if close_t.hour % funding_hours == 0 and close_t.minute == 0 and close_t != pos["entry_close_time"]:
                fcost = pos["qty_remaining"] * close * funding_rate
                equity -= fcost
                pos["funding_paid"] += fcost

            stop_hit = (low <= pos["stop_price"]) if side == 1 else (high >= pos["stop_price"])
            full_close, exit_reason = False, None

            if stop_hit:
                fill = pos["stop_price"] * (1 - side * slip)
                pnl = side * (fill - pos["entry_price"]) * pos["qty_remaining"] - abs(pos["qty_remaining"] * fill) * fee
                equity += side * (fill - pos["entry_price"]) * pos["qty_remaining"] - abs(pos["qty_remaining"] * fill) * fee
                pos["realized_pnl"] += pnl
                full_close, exit_reason = True, "stop_loss"
            else:
                if not pos["tp1_done"]:
                    tp1_hit = (high >= pos["tp1_price"]) if side == 1 else (low <= pos["tp1_price"])
                    if tp1_hit:
                        half = pos["qty_original"] * cfg["tp1_pct"]
                        fill = pos["tp1_price"] * (1 - side * slip)
                        pnl = side * (fill - pos["entry_price"]) * half - abs(half * fill) * fee
                        equity += side * (fill - pos["entry_price"]) * half - abs(half * fill) * fee
                        pos["realized_pnl"] += pnl
                        pos["qty_remaining"] -= half
                        pos["tp1_done"] = True
                        pos["tp1_time"] = close_t

                if pos["tp1_done"] and pos["qty_remaining"] > 1e-12:
                    tp2_hit = (high >= pos["tp2_price"]) if side == 1 else (low <= pos["tp2_price"])
                    if tp2_hit:
                        fill = pos["tp2_price"] * (1 - side * slip)
                        pnl = side * (fill - pos["entry_price"]) * pos["qty_remaining"] - abs(pos["qty_remaining"] * fill) * fee
                        equity += side * (fill - pos["entry_price"]) * pos["qty_remaining"] - abs(pos["qty_remaining"] * fill) * fee
                        pos["realized_pnl"] += pnl
                        full_close, exit_reason = True, "tp2"

            if full_close:
                net_pnl = pos["realized_pnl"] - pos["entry_fee"] - pos["funding_paid"]
                trades.append({
                    "entry_time": str(pos["entry_time"]), "exit_time": str(close_t),
                    "side": "long" if side == 1 else "short",
                    "entry_price": pos["entry_price"], "stop_price": pos["stop_price"],
                    "tp1_price": pos["tp1_price"], "tp2_price": pos["tp2_price"],
                    "tp1_hit": pos["tp1_done"], "net_pnl": net_pnl,
                    "exit_reason": exit_reason if not pos["tp1_done"] else ("tp1_then_" + exit_reason),
                    "regime": "ranging", "bars_held": i - pos["entry_i"],
                })
                mtm = equity
                last_full_close_i = i
                if exit_reason == "stop_loss":
                    last_stop_i_by_side[side] = i
                pos = None

        peak_equity = max(peak_equity, mtm)
        if dd_pause_enabled and not paused and peak_equity > 0 and (peak_equity - mtm) / peak_equity >= dd_pause:
            paused = True
            paused_at = close_t

        if pos is None and not paused and not np.isnan(row["atr15"]) and not np.isnan(row["adx1h"]):
            side = 1 if row["long_entry"] else (-1 if row["short_entry"] else 0)
            if side != 0 and (i - last_full_close_i) >= cd_any and (i - last_stop_i_by_side[side]) >= cd_stop_dir:
                entry_price = close * (1 + side * slip)
                stop_price = row["hh"] + cfg["atr_stop_mult"] * row["atr15"] if side == -1 else row["ll"] - cfg["atr_stop_mult"] * row["atr15"]
                stop_distance = abs(entry_price - stop_price)
                if stop_distance > 0:
                    qty_risk = (risk_pct * equity) / stop_distance
                    qty_cap = (equity * lev_cap) / entry_price
                    qty = min(qty_risk, qty_cap)
                    entry_fee = qty * entry_price * fee
                    equity -= entry_fee
                    midline = row["ll"] + 0.5 * row["range"]
                    pos = {
                        "side": side, "entry_price": entry_price, "entry_time": t,
                        "entry_i": i, "entry_close_time": close_t,
                        "stop_price": stop_price, "tp1_price": midline,
                        "tp2_price": row["hh"] if side == 1 else row["ll"],
                        "qty_original": qty, "qty_remaining": qty,
                        "tp1_done": False, "tp1_time": None,
                        "entry_fee": entry_fee, "funding_paid": 0.0, "realized_pnl": 0.0,
                    }
                    mtm = equity - entry_fee

        equity_curve.append((close_t, mtm))

    return trades, equity_curve, equity, paused, paused_at


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config_v3.yaml"
    tag = os.path.splitext(os.path.basename(config_path))[0].replace("config_", "")

    with open(os.path.join(HERE, config_path)) as f:
        cfg = yaml.safe_load(f)
    df15, df1h = load_full(cfg)
    sig = build_signals_v3(df15, df1h, cfg)

    full_start = df15.index[0] + pd.Timedelta(days=cfg["warmup_days"])
    full_end = df15.index[-1]
    total_span = (full_end - full_start)
    train_end = full_start + total_span * cfg["train_frac"]

    segments = {"train": (full_start, train_end), "oos": (train_end, full_end)}
    modes = {"edge_eval": False, "live_sim": True}  # dd_pause_enabled

    report = {"config": cfg, "tag": tag}
    for mode_name, pause_enabled in modes.items():
        report[mode_name] = {}
        for seg_name, (s, e) in segments.items():
            trades, equity_curve, final_equity, paused, paused_at = run_segment(
                sig, cfg, s, e, dd_pause_enabled=pause_enabled)
            m = compute_metrics(trades, f"{tag}_{mode_name}_{seg_name}")
            if trades:
                m.update(equity_stats(equity_curve, cfg["account_start_usdt"]))
            m["window"] = {"start": str(s), "end": str(e)}
            m["paused_by_drawdown_breaker"] = paused
            m["paused_at"] = str(paused_at) if paused_at else None
            report[mode_name][seg_name] = m
            pd.DataFrame(trades).to_csv(
                os.path.join(HERE, f"trades_{tag}_{mode_name}_{seg_name}.csv"), index=False)

    out_path = os.path.join(HERE, f"backtest_report_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
