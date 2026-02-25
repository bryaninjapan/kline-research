#!/usr/bin/env python3
"""
K-Line Research Framework — 主入口
用法：
    python run.py                                      # 使用預設設定
    python run.py --config config/btc_macd_divergence.json
    python run.py --config config/my_strategy.json
"""
import argparse
import json
import sys
from pathlib import Path

from src.data_fetcher import fetch_ohlcv
from src.indicators import add_all_indicators
from src.signal_detectors import detect_bullish_macd_divergence
from src.backtest_engine import run_backtest
from src.reporter import print_report, save_results


DEFAULT_CONFIG = "config/btc_macd_divergence.json"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def main(config_path: str = DEFAULT_CONFIG) -> None:
    print(f"\n🔧 載入設定：{config_path}")
    cfg = load_config(config_path)

    print(f"📋 策略：{cfg.get('strategy_name', 'unknown')} "
          f"{cfg.get('strategy_version', '')}")
    if cfg.get("notes"):
        print(f"📝 說明：{cfg['notes']}\n")

    # ── 1. 資料抓取 ───────────────────────────
    df = fetch_ohlcv(
        cfg["data"]["symbol"],
        cfg["data"]["interval"],
        cfg["data"]["start_date"],
        cfg["data"]["end_date"],
    )

    # ── 2. 指標計算 ───────────────────────────
    print("⚙️  計算技術指標...")
    ind = cfg["indicators"]
    df = add_all_indicators(
        df,
        macd_fast=ind.get("macd_fast", 12),
        macd_slow=ind.get("macd_slow", 26),
        macd_signal=ind.get("macd_signal", 9),
        rsi_period=ind.get("rsi_period", 14),
        atr_period=ind.get("atr_period", 14),
    )

    # ── 3. 訊號偵測 ───────────────────────────
    print("🔍 偵測底背離訊號...")
    df = detect_bullish_macd_divergence(
        df,
        swing_window=ind.get("swing_window", 5),
        lookback_bars=ind.get("divergence_lookback_bars", 60),
    )

    # ── 4. 回測 ───────────────────────────────
    print("🚀 執行回測...")
    trades = run_backtest(df, cfg)

    # ── 5. 報告 ───────────────────────────────
    print_report(trades, df, cfg)

    # ── 6. 儲存 ───────────────────────────────
    if cfg.get("output", {}).get("save_csv", True):
        save_results(trades, cfg)

    return trades, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="K-Line Trading Research Framework"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"設定檔路徑（預設：{DEFAULT_CONFIG}）",
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"❌ 找不到設定檔：{args.config}")
        print(f"   可用設定：{list(Path('config').glob('*.json'))}")
        sys.exit(1)

    main(args.config)
