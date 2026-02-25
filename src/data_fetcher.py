"""
資料抓取模組 — Binance 免費 OHLCV API（無需 API Key）
支援所有 Binance 現貨交易對，週期：1m 5m 15m 1h 4h 1d 1w
"""
import time
import requests
import pandas as pd
from datetime import datetime, timezone


def fetch_ohlcv(symbol: str, interval: str,
                start_date: str, end_date: str) -> pd.DataFrame:
    """
    從 Binance 抓取 OHLCV 歷史資料。

    Args:
        symbol:     交易對，例如 'BTCUSDT', 'ETHUSDT'
        interval:   K 棒週期，例如 '4h', '1h', '1d'
        start_date: 開始日期，格式 'YYYY-MM-DD'
        end_date:   結束日期，格式 'YYYY-MM-DD'

    Returns:
        DataFrame，index 為 UTC 時間戳記，欄位為 open/high/low/close/volume
    """
    print(f"📡 Fetching {symbol} {interval}  {start_date} → {end_date}")

    url = "https://api.binance.com/api/v3/klines"
    start_ts = _to_ms(start_date)
    end_ts = _to_ms(end_date)

    all_klines = []
    current_ts = start_ts

    while current_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_ts,
            "endTime": end_ts,
            "limit": 1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            klines = resp.json()
        except Exception as exc:
            print(f"⚠️  API error: {exc} — retrying in 3s")
            time.sleep(3)
            continue

        if not klines:
            break

        all_klines.extend(klines)
        current_ts = klines[-1][0] + 1

        if len(klines) < 1000:
            break

        time.sleep(0.12)   # Binance rate limit 緩衝

    if not all_klines:
        raise ValueError(f"No data returned for {symbol} {interval}")

    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep="first")].sort_index()

    print(f"✅ {len(df):,} candles  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def _to_ms(date_str: str) -> int:
    return int(
        datetime.strptime(date_str, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp() * 1000
    )
