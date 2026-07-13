"""Event-driven 15m backtest for the EIGEN/USDT v1 (raw) breakout+RSI strategy.

Position sizing: each new trade risks the FULL current equity as margin
(no per-trade risk %, since the spec didn't define one) -> notional = equity * leverage.
This is disclosed explicitly in the report since it drives strong compounding.

Exit priority when more than one condition fires on the same bar:
  hard stop (20% margin) > ATR stop (initial/trailing) > reversal take-profit.
The ATR trailing stop is checked against the bar's ADVERSE extreme BEFORE being
tightened by this same bar's FAVORABLE extreme, so a position can never be
stopped out by a level only just set within the same bar.
"""
import json
import os

import numpy as np
import pandas as pd
import yaml

from strategy import build_signals

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def load(cfg):
    symbol = cfg["symbol"]
    df15 = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}_15m.csv"), index_col=0, parse_dates=True)
    df1h = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}_1h.csv"), index_col=0, parse_dates=True)
    df4h = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}_4h.csv"), index_col=0, parse_dates=True)
    return df15, df1h, df4h


def run_backtest(sig, cfg):
    lev = cfg["leverage"]
    fee = cfg["fee_pct"]
    slip = cfg["slippage_pct"]
    atr_mult = cfg["atr_stop_mult"]
    hard_stop_pct = cfg["hard_stop_margin_pct"]
    funding_rate = cfg["funding_rate_per_interval"]
    funding_hours = cfg["funding_interval_hours"]

    equity = float(cfg["account_start_usdt"])
    pos = None  # dict when open
    trades = []
    equity_curve = []  # (close_time, mark_to_market_equity)

    idx = sig.index
    for i in range(len(idx)):
        row = sig.iloc[i]
        t = idx[i]
        close_t = t + pd.Timedelta(minutes=15)
        close, high, low = row["close"], row["high"], row["low"]

        if pos is None:
            mtm = equity
        else:
            side = pos["side"]
            unreal = side * (close - pos["entry_price"]) * pos["qty"]
            mtm = equity + unreal

            if close_t.hour % funding_hours == 0 and close_t.minute == 0 and close_t != pos["entry_close_time"]:
                fcost = pos["qty"] * close * funding_rate
                equity -= fcost
                pos["funding_paid"] += fcost

            worst_price = low if side == 1 else high
            worst_margin_loss = -side * (worst_price - pos["entry_price"]) * pos["qty"] / pos["margin"]
            pos["worst_margin_loss_pct"] = max(pos["worst_margin_loss_pct"], worst_margin_loss)
            hard_stop_price = pos["entry_price"] - side * (hard_stop_pct * pos["margin"]) / pos["qty"]
            hard_stop_hit = (low <= hard_stop_price) if side == 1 else (high >= hard_stop_price)

            atr_stop_hit = (low <= pos["trailing_stop"]) if side == 1 else (high >= pos["trailing_stop"])

            reversal_hit, reversal_reason = False, None
            unreal_at_close = side * (close - pos["entry_price"]) * pos["qty"]
            if unreal_at_close > 0:
                ll_now, range_now = row["ll"], row["range"]
                midline = ll_now + cfg["take_profit_midline_pct"] * range_now
                if side == 1:
                    opposite = ll_now + cfg["range_top_pct"] * range_now
                    rsi_cross = pos["prev_rsi"] is not None and pos["prev_rsi"] >= 50 and row["rsi"] < 50
                    if close <= midline:
                        reversal_hit, reversal_reason = True, "reversal_midline"
                    elif close >= opposite:
                        reversal_hit, reversal_reason = True, "reversal_opposite"
                    elif rsi_cross:
                        reversal_hit, reversal_reason = True, "reversal_rsi50"
                else:
                    opposite = ll_now + cfg["range_bottom_pct"] * range_now
                    rsi_cross = pos["prev_rsi"] is not None and pos["prev_rsi"] <= 50 and row["rsi"] > 50
                    if close >= midline:
                        reversal_hit, reversal_reason = True, "reversal_midline"
                    elif close <= opposite:
                        reversal_hit, reversal_reason = True, "reversal_opposite"
                    elif rsi_cross:
                        reversal_hit, reversal_reason = True, "reversal_rsi50"

            exit_price, exit_reason = None, None
            if hard_stop_hit:
                exit_price = hard_stop_price * (1 - side * slip)
                exit_reason = "hard_stop"
            elif atr_stop_hit:
                exit_price = pos["trailing_stop"] * (1 - side * slip)
                exit_reason = "atr_stop"
            elif reversal_hit:
                exit_price = close * (1 - side * slip)
                exit_reason = reversal_reason

            if exit_reason is not None:
                exit_fee = abs(pos["qty"] * exit_price) * fee
                net_pnl = side * (exit_price - pos["entry_price"]) * pos["qty"] - pos["entry_fee"] - exit_fee - pos["funding_paid"]
                equity += side * (exit_price - pos["entry_price"]) * pos["qty"] - exit_fee
                trades.append({
                    "entry_time": str(pos["entry_time"]), "exit_time": str(close_t),
                    "side": "long" if side == 1 else "short",
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "qty": pos["qty"], "margin": pos["margin"],
                    "net_pnl": net_pnl, "exit_reason": exit_reason,
                    "regime": pos["regime"], "bars_held": i - pos["entry_i"],
                    "worst_margin_loss_pct": pos["worst_margin_loss_pct"],
                })
                mtm = equity
                pos = None
            else:
                if side == 1:
                    pos["extreme"] = max(pos["extreme"], high)
                    candidate = pos["extreme"] - atr_mult * row["atr4h"]
                    pos["trailing_stop"] = max(pos["trailing_stop"], candidate)
                else:
                    pos["extreme"] = min(pos["extreme"], low)
                    candidate = pos["extreme"] + atr_mult * row["atr4h"]
                    pos["trailing_stop"] = min(pos["trailing_stop"], candidate)
                pos["prev_rsi"] = row["rsi"]

        if pos is None and not np.isnan(row["atr4h"]) and not np.isnan(row["adx1h"]):
            side = 1 if row["long_entry"] else (-1 if row["short_entry"] else 0)
            if side != 0:
                entry_price = close * (1 + side * slip)
                margin = equity
                notional = margin * lev
                qty = notional / entry_price
                entry_fee = notional * fee
                equity -= entry_fee
                init_stop = entry_price - side * atr_mult * row["atr4h"]
                pos = {
                    "side": side, "entry_price": entry_price, "qty": qty, "margin": margin,
                    "entry_fee": entry_fee, "funding_paid": 0.0,
                    "trailing_stop": init_stop, "extreme": entry_price,
                    "entry_time": t, "entry_i": i, "entry_close_time": close_t,
                    "regime": "trending" if row["adx1h"] >= cfg["adx_trend_threshold"] else "ranging",
                    "prev_rsi": row["rsi"], "worst_margin_loss_pct": 0.0,
                }
                mtm = equity - entry_fee + 0  # unrealized 0 at entry bar

        equity_curve.append((close_t, mtm))

    return trades, equity_curve, equity


def compute_metrics(trades, label=""):
    n = len(trades)
    if n == 0:
        return {"label": label, "trades": 0}
    pnls = [t["net_pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return {
        "label": label,
        "trades": n,
        "win_rate": len(wins) / n,
        "total_pnl_usdt": total_pnl,
        "avg_win_usdt": avg_win,
        "avg_loss_usdt": avg_loss,
        "payoff_ratio": (avg_win / abs(avg_loss)) if avg_loss != 0 else float("nan"),
        "profit_factor": (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("nan"),
    }


def equity_stats(equity_curve, start_equity):
    df = pd.DataFrame(equity_curve, columns=["t", "equity"]).set_index("t")
    daily = df["equity"].resample("1D").last().ffill()
    rets = daily.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(365)) if rets.std() > 0 else float("nan")
    cummax = df["equity"].cummax()
    dd = (df["equity"] / cummax - 1)
    max_dd = dd.min()
    final_equity = df["equity"].iloc[-1]
    total_return_pct = (final_equity / start_equity - 1) * 100
    return {
        "sharpe_annualized": float(sharpe),
        "max_drawdown_pct": float(max_dd * 100),
        "final_equity_usdt": float(final_equity),
        "total_return_pct": float(total_return_pct),
    }


def main():
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    df15, df1h, df4h = load(cfg)
    sig = build_signals(df15, df1h, df4h, cfg)

    cutoff = sig.index[-1] - pd.Timedelta(days=cfg["backtest_days"])
    sig_bt = sig[sig.index >= cutoff].copy()

    trades, equity_curve, final_equity = run_backtest(sig_bt, cfg)

    overall = compute_metrics(trades, "overall")
    overall.update(equity_stats(equity_curve, cfg["account_start_usdt"]))
    trending = compute_metrics([t for t in trades if t["regime"] == "trending"], "trending")
    ranging = compute_metrics([t for t in trades if t["regime"] == "ranging"], "ranging")

    hard_stop_trades = [t for t in trades if t["exit_reason"] == "hard_stop"]
    worst_losses = sorted((t["worst_margin_loss_pct"] for t in trades), reverse=True)[:10]
    near_liq = [t for t in trades if t["worst_margin_loss_pct"] >= 0.5]

    report = {
        "config": cfg,
        "backtest_window": {"start": str(sig_bt.index[0]), "end": str(sig_bt.index[-1])},
        "overall": overall,
        "by_regime": {"trending": trending, "ranging": ranging},
        "hard_stop": {
            "count": len(hard_stop_trades),
            "total_trades": len(trades),
            "near_liquidation_ge_50pct_margin_loss_count": len(near_liq),
            "worst_10_margin_loss_pct": worst_losses,
        },
    }

    with open(os.path.join(HERE, "backtest_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    pd.DataFrame(trades).to_csv(os.path.join(HERE, "trades.csv"), index=False)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
