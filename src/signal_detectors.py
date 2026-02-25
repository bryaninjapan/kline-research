"""
訊號偵測模組
實作各種技術形態偵測，全部處理「無未來資訊洩漏」問題。

底背離偵測流程：
  1. 找到 swing low（需要右側 N 棒確認 → 延遲 N 棒）
  2. 比對兩個相鄰 swing low 的價格與 MACD
  3. 訊號只在「確認棒」之後才可使用
"""
import numpy as np
import pandas as pd
from .indicators import swing_lows, swing_highs


def detect_bullish_macd_divergence(df: pd.DataFrame,
                                    swing_window: int = 5,
                                    lookback_bars: int = 60) -> pd.DataFrame:
    """
    偵測 MACD 底背離（Bullish Divergence）。

    定義：
      - 價格形成 Lower Low（更低的低點）
      - 同期 MACD 線形成 Higher Low（更高的低點）
      → 下跌動能衰竭訊號

    訊號欄位（加到 df）：
      div_signal:       bool，該棒是否為底背離的 swing low
      div_low_price:    float，底背離低點價格（作為止損參考）
      div_confirmed_pos:int，訊號在第幾個 bar index 才可使用（防洩漏）

    Args:
        df:             含 close, macd_line 欄位的 DataFrame
        swing_window:   swing low 確認所需的右側棒數
        lookback_bars:  回溯多少棒尋找配對的前一個 swing low

    Returns:
        添加了上述三個欄位的 DataFrame
    """
    df = df.copy()
    close = df["close"]
    macd_line = df["macd_line"]

    sl_mask = swing_lows(close, swing_window)
    sl_idx_list = df.index[sl_mask].tolist()

    df["div_signal"] = False
    df["div_low_price"] = np.nan
    df["div_confirmed_pos"] = -1

    pos_of = {idx: i for i, idx in enumerate(df.index)}

    for i, curr_idx in enumerate(sl_idx_list):
        if i == 0:
            continue

        curr_pos = pos_of[curr_idx]
        curr_price = close[curr_idx]
        curr_macd = macd_line[curr_idx]

        # 在 lookback 範圍內找前一個 swing low
        candidates = [
            p for p in sl_idx_list[:i]
            if pos_of[p] >= curr_pos - lookback_bars
        ]
        if not candidates:
            continue

        prev_idx = candidates[-1]
        prev_price = close[prev_idx]
        prev_macd = macd_line[prev_idx]

        # 底背離條件
        if curr_price < prev_price and curr_macd > prev_macd:
            df.at[curr_idx, "div_signal"] = True
            df.at[curr_idx, "div_low_price"] = curr_price
            confirmed_pos = min(curr_pos + swing_window, len(df) - 1)
            df.at[curr_idx, "div_confirmed_pos"] = confirmed_pos

    n_signals = df["div_signal"].sum()
    print(f"   偵測到 {n_signals} 個 MACD 底背離訊號")
    return df


def detect_bearish_macd_divergence(df: pd.DataFrame,
                                    swing_window: int = 5,
                                    lookback_bars: int = 60) -> pd.DataFrame:
    """
    偵測 MACD 頂背離（Bearish Divergence）。

    定義：
      - 價格形成 Higher High
      - MACD 線形成 Lower High
      → 上漲動能衰竭訊號

    欄位命名與底背離一致（div_signal, div_low_price → 改用 div_high_price）
    """
    df = df.copy()
    close = df["close"]
    macd_line = df["macd_line"]

    sh_mask = swing_highs(close, swing_window)
    sh_idx_list = df.index[sh_mask].tolist()

    df["bear_div_signal"] = False
    df["div_high_price"] = np.nan
    df["bear_div_confirmed_pos"] = -1

    pos_of = {idx: i for i, idx in enumerate(df.index)}

    for i, curr_idx in enumerate(sh_idx_list):
        if i == 0:
            continue

        curr_pos = pos_of[curr_idx]
        curr_price = close[curr_idx]
        curr_macd = macd_line[curr_idx]

        candidates = [
            p for p in sh_idx_list[:i]
            if pos_of[p] >= curr_pos - lookback_bars
        ]
        if not candidates:
            continue

        prev_idx = candidates[-1]
        prev_price = close[prev_idx]
        prev_macd = macd_line[prev_idx]

        if curr_price > prev_price and curr_macd < prev_macd:
            df.at[curr_idx, "bear_div_signal"] = True
            df.at[curr_idx, "div_high_price"] = curr_price
            confirmed_pos = min(curr_pos + swing_window, len(df) - 1)
            df.at[curr_idx, "bear_div_confirmed_pos"] = confirmed_pos

    n_signals = df["bear_div_signal"].sum()
    print(f"   偵測到 {n_signals} 個 MACD 頂背離訊號")
    return df
