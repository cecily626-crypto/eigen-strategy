# EIGEN/USDT strategy (v1 raw)

Single-symbol breakout+RSI reversion strategy. Data from Binance spot klines,
intended execution venue is LBank's EIGENUSDT USDT-margined perpetual.

- `config.yaml` — every parameter (nothing hardcoded in the strategy/backtest code)
- `fetch_data.py` — pulls 15m/1h/4h EIGENUSDT klines from Binance
- `indicators.py` — ema/rsi/atr/adx/donchian, shift(1)-safe
- `strategy.py` — entry-signal construction, leak-free cross-timeframe alignment
- `backtest.py` — event-driven backtest (20x leverage, fees, funding, slippage)
- `validate_no_lookahead.py` — automated proof the signals use no future data

Run order: `python3 fetch_data.py && python3 backtest.py && python3 validate_no_lookahead.py`
