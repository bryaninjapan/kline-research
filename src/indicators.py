"""
技術指標計算模組
全部使用純 pandas/numpy，無需安裝 ta-lib
所有指標均為「無未來資訊洩漏」實作
"""
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 趨勢類
# ──────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    """指數移動平均（EMA）"""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """簡單移動平均（SMA）"""
    return series.rolling(period).mean()


def macd(close: pd.Series,
         fast: int = 12, slow: int = 26, signal: int = 9
         ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD 指標
    Returns:
        (macd_line, signal_line, histogram)
    """
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


# ──────────────────────────────────────────────
# 動能類
# ──────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI（相對強弱指標）
    使用 Wilder 平滑法（ewm com = period-1）
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def stoch_rsi(close: pd.Series,
              rsi_period: int = 14, stoch_period: int = 14,
              k_smooth: int = 3, d_smooth: int = 3
              ) -> tuple[pd.Series, pd.Series]:
    """Stochastic RSI，返回 (K, D)"""
    r = rsi(close, rsi_period)
    lo = r.rolling(stoch_period).min()
    hi = r.rolling(stoch_period).max()
    k = 100 * (r - lo) / (hi - lo + 1e-10)
    d = k.rolling(d_smooth).mean()
    return k.rolling(k_smooth).mean(), d


# ──────────────────────────────────────────────
# 波動率類
# ──────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR（平均真實波動幅度）"""
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat([
        hi - lo,
        (hi - cl.shift(1)).abs(),
        (lo - cl.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def bollinger_bands(close: pd.Series,
                    period: int = 20, std_mult: float = 2.0
                    ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """布林通道，返回 (upper, middle, lower)"""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + std_mult * std, mid, mid - std_mult * std


# ──────────────────────────────────────────────
# Swing 高低點偵測（無未來資訊洩漏）
# ──────────────────────────────────────────────

def swing_lows(price: pd.Series, window: int = 5) -> pd.Series:
    """
    偵測 Swing Low（局部最低點）。
    一個棒位被標記為 swing low 的條件：
    它是 [i-window, i+window] 範圍內的最低點。

    ⚠️  注意：此函式本身包含未來資訊（需要右側 window 根棒）。
    在回測引擎中，訊號需延遲 `window` 根棒才能使用。
    見 signal_detectors.py 的防洩漏處理。
    """
    result = pd.Series(False, index=price.index)
    arr = price.values
    n = len(arr)
    for i in range(window, n - window):
        segment = arr[i - window: i + window + 1]
        if arr[i] == segment.min() and arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            result.iloc[i] = True
    return result


def swing_highs(price: pd.Series, window: int = 5) -> pd.Series:
    """偵測 Swing High（局部最高點），注意事項同 swing_lows。"""
    result = pd.Series(False, index=price.index)
    arr = price.values
    n = len(arr)
    for i in range(window, n - window):
        segment = arr[i - window: i + window + 1]
        if arr[i] == segment.max() and arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            result.iloc[i] = True
    return result


# ──────────────────────────────────────────────
# 一次性計算所有常用指標（給 DataFrame 加欄位）
# ──────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame,
                       macd_fast=12, macd_slow=26, macd_signal=9,
                       rsi_period=14, atr_period=14) -> pd.DataFrame:
    """
    在 DataFrame 上直接加入所有常用指標欄位。
    df 必須含有 open, high, low, close, volume 欄位。
    """
    df = df.copy()
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd(
        df["close"], macd_fast, macd_slow, macd_signal
    )
    df["rsi"] = rsi(df["close"], rsi_period)
    df["atr"] = atr(df, atr_period)
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    bb_up, bb_mid, bb_lo = bollinger_bands(df["close"])
    df["bb_upper"] = bb_up
    df["bb_mid"] = bb_mid
    df["bb_lower"] = bb_lo
    return df
