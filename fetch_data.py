"""Pull Binance spot klines for EIGENUSDT and cache them as CSV under data/.

Fetches backtest_days + warmup_days of history so indicators (ATR/RSI/ADX)
are fully warmed up before the first bar actually used in the backtest.
"""
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd
import yaml

BASE = "https://api.binance.com/api/v3/klines"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def _get(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "eigen-strategy/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1.5)


def fetch_klines(symbol, interval, start_ms, end_ms):
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        q = urllib.parse.urlencode({
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms, "limit": 1000,
        })
        batch = _get(f"{BASE}?{q}")
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        if len(batch) < 1000:
            break
        cursor = last_open + 1
        time.sleep(0.3)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_vol", "trades", "taker_base_vol", "taker_quote_vol", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.drop_duplicates("open_time").sort_values("open_time")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.set_index("date")[["open", "high", "low", "close", "volume"]]


def main():
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    symbol = cfg["symbol"]
    total_days = cfg["backtest_days"] + cfg["warmup_days"]
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - total_days * 86400 * 1000

    os.makedirs(DATA_DIR, exist_ok=True)
    for interval, tag in [("15m", "15m"), ("1h", "1h"), ("4h", "4h")]:
        df = fetch_klines(symbol, interval, start_ms, end_ms)
        path = os.path.join(DATA_DIR, f"{symbol}_{tag}.csv")
        df.to_csv(path)
        print(f"{tag}: {len(df)} bars, {df.index[0]} -> {df.index[-1]}  saved to {path}")


if __name__ == "__main__":
    main()
