"""Pull the FULL available Binance history for EIGENUSDT (15m + 1h), not just
a fixed lookback window -- auto-detects the symbol's listing date."""
import os
import time

import pandas as pd
import yaml

from fetch_data import BASE, DATA_DIR, HERE, _get, fetch_klines


def find_listing_date(symbol):
    q = f"{BASE}?symbol={symbol}&interval=1d&startTime=0&limit=1"
    row = _get(q)[0]
    return row[0]  # open_time ms


def main():
    with open(os.path.join(HERE, "config_v3.yaml")) as f:
        cfg = yaml.safe_load(f)
    symbol = cfg["symbol"]
    start_ms = find_listing_date(symbol)
    end_ms = int(time.time() * 1000)
    print(f"listing date detected: {pd.to_datetime(start_ms, unit='ms', utc=True)}")

    os.makedirs(DATA_DIR, exist_ok=True)
    for interval, tag in [("15m", "15m"), ("1h", "1h")]:
        df = fetch_klines(symbol, interval, start_ms, end_ms)
        path = os.path.join(DATA_DIR, f"{symbol}_{tag}_full.csv")
        df.to_csv(path)
        print(f"{tag}: {len(df)} bars, {df.index[0]} -> {df.index[-1]}  saved to {path}")


if __name__ == "__main__":
    main()
