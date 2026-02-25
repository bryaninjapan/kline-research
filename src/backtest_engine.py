"""
回測引擎（通用狀態機）
支援多種止損類型，完整處理「無未來資訊洩漏」

止損類型：
  divergence_low  → 底背離低點（推薦）
  prev_candle_low → 入場前一棒最低點
  fixed_pct       → 固定百分比止損
  atr_multiple    → ATR 倍數止損
"""
import numpy as np
import pandas as pd


def run_backtest(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    執行回測。

    Args:
        df:  含 OHLCV + 指標 + 訊號欄位的 DataFrame
        cfg: 策略設定（從 JSON config 載入）

    Returns:
        交易記錄 DataFrame
    """
    # ── 設定解包 ──────────────────────────────
    entry_cfg = cfg["entry"]
    exit_cfg = cfg["exit"]
    cost_cfg = cfg["costs"]

    rebound_pct = entry_cfg.get("rebound_pct", 2.0)
    rsi_min = entry_cfg.get("rsi_min", 30)
    rsi_max = entry_cfg.get("rsi_max", 40)

    tp_rsi = exit_cfg.get("tp_rsi", 70)
    sl_type = exit_cfg.get("sl_type", "divergence_low")
    sl_buffer = exit_cfg.get("sl_buffer_pct", 0.1) / 100
    sl_pct = abs(exit_cfg.get("sl_pct", 5.0)) / 100
    sl_atr_mult = exit_cfg.get("sl_atr_mult", 2.0)
    time_stop = exit_cfg.get("time_stop_bars", 30)

    fee = cost_cfg.get("fee_pct", 0.06) / 100
    slip = cost_cfg.get("slippage_pct", 0.05) / 100
    total_cost = fee + slip

    # ── 陣列化（速度最佳化）──────────────────
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    rsi_arr = df["rsi"].values
    atr_arr = df["atr"].values if "atr" in df.columns else np.zeros(len(df))
    div_signal = df["div_signal"].values
    div_low = df["div_low_price"].values
    div_conf_pos = df["div_confirmed_pos"].values

    # ── 建立確認位置索引 ──────────────────────
    conf_map: dict[int, list] = {}
    for i in range(len(df)):
        if div_signal[i]:
            cp = int(div_conf_pos[i])
            if cp >= 0:
                conf_map.setdefault(cp, []).append({
                    "div_low": div_low[i],
                    "div_bar": i,
                })

    # ── 主回測迴圈 ────────────────────────────
    trades = []
    in_pos = False
    entry_price = stop_loss = entry_bar = 0
    pending = []   # 已確認但尚未入場的訊號

    for i in range(1, len(df)):

        # 加入新確認訊號
        if i in conf_map:
            pending.extend(conf_map[i])

        if not in_pos:
            for sig in pending[:]:
                d_low = sig["div_low"]
                threshold = d_low * (1 + rebound_pct / 100)

                if close[i] >= threshold and rsi_min <= rsi_arr[i] <= rsi_max:
                    # ── 計算止損位 ────────────
                    if sl_type == "divergence_low":
                        sl = d_low * (1 - sl_buffer)
                    elif sl_type == "prev_candle_low":
                        sl = low[i - 1] * (1 - sl_buffer)
                    elif sl_type == "fixed_pct":
                        sl = close[i] * (1 - sl_pct)
                    elif sl_type == "atr_multiple":
                        sl = close[i] - sl_atr_mult * atr_arr[i]
                    else:
                        sl = d_low * (1 - sl_buffer)

                    # ── 入場 ─────────────────
                    in_pos = True
                    entry_price = close[i] * (1 + total_cost)
                    stop_loss = sl
                    entry_bar = i

                    trades.append({
                        "entry_idx": i,
                        "entry_time": df.index[i],
                        "entry_price": entry_price,
                        "entry_raw": close[i],
                        "stop_loss": stop_loss,
                        "div_low": d_low,
                        "entry_rsi": rsi_arr[i],
                        "sl_pct": (entry_price - stop_loss) / entry_price * 100,
                    })
                    pending = []
                    break

        else:
            bars = i - entry_bar
            reason = None
            exit_px = None

            # 1. 止損（intrabar low 觸及）
            if low[i] <= stop_loss:
                reason = "stop_loss"
                exit_px = stop_loss * (1 - total_cost)

            # 2. 時間止損
            elif bars >= time_stop:
                reason = "time_stop"
                exit_px = close[i] * (1 - total_cost)

            # 3. 止盈（RSI 達標）
            elif rsi_arr[i] >= tp_rsi:
                reason = "take_profit"
                exit_px = close[i] * (1 - total_cost)

            if reason:
                in_pos = False
                pnl = (exit_px - entry_price) / entry_price * 100
                trades[-1].update({
                    "exit_idx": i,
                    "exit_time": df.index[i],
                    "exit_price": exit_px,
                    "exit_raw": close[i],
                    "exit_reason": reason,
                    "bars_held": bars,
                    "hours_held": bars * _interval_hours(cfg),
                    "pnl_pct": pnl,
                    "win": pnl > 0,
                    "year": df.index[i].year,
                })
                pending = []

    return pd.DataFrame(trades)


def _interval_hours(cfg: dict) -> float:
    interval_map = {
        "1m": 1/60, "5m": 5/60, "15m": 0.25,
        "1h": 1, "4h": 4, "1d": 24, "1w": 168,
    }
    return interval_map.get(cfg["data"]["interval"], 4)
