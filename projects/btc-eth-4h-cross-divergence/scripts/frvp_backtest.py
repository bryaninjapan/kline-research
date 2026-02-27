#!/usr/bin/env python3
"""
FRVP Cross-Asset Divergence Backtest  (frvp_backtest.py)
=========================================================
Rules confirmed 2026-02-27:

DETECTION (Bearish — short only)
  - BTC and ETH each form two consecutive swing highs 4–12 bars apart
  - One asset HH, other LH; |price diff| > 1%
  - SH2 of both assets within ±sync_tolerance (3) bars of each other

FRVP PROXY
  - Range : last swing low (swing_window=5) before SH1 → SH2
            (max lookback = max_frvp_lookback = 100 bars)
  - POC   : VWAP of the range  (volume-weighted avg close)
  - VAH   : 85th percentile of closes in the range

ENTRY
  - Signal confirmed at max(SH2_btc, SH2_eth) + swing_window (no look-ahead)
  - Signal expires after signal_expiry_bars (30) if POC not broken
  - First asset closes below its POC → watch for second within poc_break_bars (3)
  - Both confirmed → short entry at OPEN of next bar

STOP LOSS  (stop_mode: vah | prev_high | sh2)
  - vah      : stop = VAH * (1 + vah_buffer_pct=0.5%)
  - prev_high: stop = high[entry_bar-1] * (1 + vah_buffer_pct=0.5%)  [default]
  - sh2      : stop = SH2_price * (1 + vah_buffer_pct=0.5%)
  - Triggered when high[i] >= stop (either asset → exit both)

TAKE PROFIT (RSI-based, no time stop)
  - BTC 4H RSI (period=14) first closes < 30  → rsi_countdown = 1
  - If next bar RSI still < 30 → exit at OPEN of that bar (2nd candle)
  - If RSI bounces back ≥ 30 → reset countdown to 0

COST
  - fee_pct = 0.06% per side
  - PnL = equal-weight average of BTC and ETH

USAGE
  python scripts/frvp_backtest.py
  python scripts/frvp_backtest.py --swing-window 4 --min-gap 3 --max-gap 15
"""

import argparse
import sys
import time
import requests
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
OUTPUT_DIR  = ROOT / "output"
RESULTS_DIR = ROOT / "results"

# ── Default Config ─────────────────────────────────────────────────────────
DEFAULT_CFG = {
    # Divergence detection
    "swing_window":       5,     # bars each side to confirm swing high/low
    "min_div_gap":        4,     # min bars between SH1 and SH2 (each asset)
    "max_div_gap":        12,    # max bars between SH1 and SH2
    "min_price_diff_pct": 1.0,   # min % diff between SH1 and SH2 prices
    "sync_tolerance":     3,     # SH2 of both assets within ±N bars
    # FRVP
    "max_frvp_lookback":  100,   # max bars before SH1 to search for swing low
    # Entry / signal lifecycle
    "signal_expiry_bars": 30,    # signal expires N bars after confirmation
    "poc_break_bars":     3,     # wait N bars for second asset to break POC
    # Stop
    "vah_buffer_pct":     0.5,   # stop buffer %
    "stop_mode":          "prev_high",  # vah | prev_high | sh2
    # Take profit
    "tp_rsi_period":      14,
    "tp_rsi_threshold":   30,    # exit when BTC RSI closes < 30 (2nd bar)
    # Cost
    "fee_pct":            0.06,  # per side
}


# ══════════════════════════════════════════════════════════════════════════
# 1. Data  (Binance API — same source as MACD backtest scripts)
# ══════════════════════════════════════════════════════════════════════════

def fetch_binance_ohlcv(symbol: str, interval: str,
                        start_date: str, end_date: str) -> pd.DataFrame:
    print(f"  📡 Fetching {symbol} {interval} {start_date} → {end_date} from Binance...")
    url      = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts   = int(datetime.strptime(end_date, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)

    all_klines   = []
    current_ts   = start_ts

    while current_ts < end_ts:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": current_ts, "endTime": end_ts, "limit": 1000}
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            klines = resp.json()
        except Exception as e:
            print(f"    ⚠️  API error: {e}, retrying...")
            time.sleep(3)
            continue
        if not klines:
            break
        all_klines.extend(klines)
        current_ts = klines[-1][0] + 1
        if len(klines) < 1000:
            break
        time.sleep(0.12)

    if not all_klines:
        raise ValueError(f"No data fetched for {symbol}")

    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "tb_base", "tb_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    print(f"     ✅ {len(df):,} candles  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ══════════════════════════════════════════════════════════════════════════
# 2. Indicators
# ══════════════════════════════════════════════════════════════════════════

def calc_rsi(close: pd.Series, period: int = 14) -> np.ndarray:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag    = gain.ewm(com=period - 1, adjust=False).mean()
    al    = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = ag / (al + 1e-10)
    return (100 - 100 / (1 + rs)).values


# ══════════════════════════════════════════════════════════════════════════
# 3. Swing detection
# ══════════════════════════════════════════════════════════════════════════

def find_swing_highs(price: np.ndarray, window: int) -> list:
    """
    Returns list of (pos, price).
    Swing high: strictly higher than all bars in [i-window, i+window].
    Requires `window` future bars to confirm — safe against look-ahead.
    """
    result = []
    n = len(price)
    for i in range(window, n - window):
        segment = price[i - window: i + window + 1]
        if price[i] == segment.max() and price[i] > price[i - 1] and price[i] > price[i + 1]:
            result.append((i, price[i]))
    return result


def find_swing_lows(price: np.ndarray, window: int) -> list:
    """Returns list of (pos, price) for swing lows."""
    result = []
    n = len(price)
    for i in range(window, n - window):
        segment = price[i - window: i + window + 1]
        if price[i] == segment.min() and price[i] < price[i - 1] and price[i] < price[i + 1]:
            result.append((i, price[i]))
    return result


# ══════════════════════════════════════════════════════════════════════════
# 4. FRVP proxy
# ══════════════════════════════════════════════════════════════════════════

def calc_frvp_proxy(close: np.ndarray, volume: np.ndarray,
                    start: int, end: int) -> tuple:
    """
    POC = VWAP(close) over [start, end]
    VAH = 85th percentile of closes over [start, end]
    Both are proxies for the Fixed Range Volume Profile levels.
    """
    c = close[start: end + 1]
    v = volume[start: end + 1]
    if len(c) == 0:
        p = float(close[end])
        return p, p
    total_v = v.sum()
    vwap    = float((c * v).sum() / total_v) if total_v > 0 else float(c.mean())
    vah     = float(np.percentile(c, 85))
    return vwap, max(vah, vwap)   # ensure VAH >= POC


# ══════════════════════════════════════════════════════════════════════════
# 5. Divergence detection
# ══════════════════════════════════════════════════════════════════════════

def detect_divergences(df_btc: pd.DataFrame, df_eth: pd.DataFrame,
                       cfg: dict) -> list:
    """
    Detects BEARISH cross-asset divergences (for short trades):
      One asset makes HH, the other LH, within ±sync_tolerance bars.

    Returns list of signal dicts sorted by confirmed_pos.
    confirmed_pos = max(SH2_btc_pos, SH2_eth_pos) + swing_window.
    This is the earliest bar at which the signal may be acted upon
    (prevents look-ahead bias — both swing highs need window bars to confirm).
    """
    sw      = cfg["swing_window"]
    min_gap = cfg["min_div_gap"]
    max_gap = cfg["max_div_gap"]
    min_pct = cfg["min_price_diff_pct"]
    sync    = cfg["sync_tolerance"]
    max_lb  = cfg["max_frvp_lookback"]
    n       = len(df_btc)

    btc_h = df_btc["high"].values
    eth_h = df_eth["high"].values
    btc_l = df_btc["low"].values
    eth_l = df_eth["low"].values
    btc_c = df_btc["close"].values
    eth_c = df_eth["close"].values
    btc_v = df_btc["volume"].values
    eth_v = df_eth["volume"].values

    btc_sh = find_swing_highs(btc_h, sw)
    eth_sh = find_swing_highs(eth_h, sw)
    btc_sl = [p for p, _ in find_swing_lows(btc_l, sw)]
    eth_sl = [p for p, _ in find_swing_lows(eth_l, sw)]

    def last_sl(sl_list, ref_pos):
        """Most recent swing low position strictly before ref_pos, within max_lb."""
        cands = [p for p in sl_list if p < ref_pos and p >= ref_pos - max_lb]
        return cands[-1] if cands else max(0, ref_pos - max_lb)

    signals        = []
    seen_conf_pos  = set()   # one signal per confirmed bar

    for b_i in range(1, len(btc_sh)):
        b1_pos, b1_p = btc_sh[b_i - 1]
        b2_pos, b2_p = btc_sh[b_i]

        btc_gap = b2_pos - b1_pos
        if not (min_gap <= btc_gap <= max_gap):
            continue
        if abs(b2_p - b1_p) / b1_p * 100 < min_pct:
            continue
        btc_hh = b2_p > b1_p

        for e_i in range(1, len(eth_sh)):
            e1_pos, e1_p = eth_sh[e_i - 1]
            e2_pos, e2_p = eth_sh[e_i]

            eth_gap = e2_pos - e1_pos
            if not (min_gap <= eth_gap <= max_gap):
                continue
            if abs(e2_p - e1_p) / e1_p * 100 < min_pct:
                continue
            eth_hh = e2_p > e1_p

            # Must form opposite patterns (one HH, one LH)
            if btc_hh == eth_hh:
                continue

            # SH2 of both assets must be close in time
            if abs(b2_pos - e2_pos) > sync:
                continue

            confirmed_pos = max(b2_pos, e2_pos) + sw
            if confirmed_pos >= n:
                continue
            if confirmed_pos in seen_conf_pos:
                continue   # keep only first divergence per bar

            # ── FRVP for BTC ──────────────────────────────────────
            btc_start = last_sl(btc_sl, b1_pos)
            btc_poc, btc_vah = calc_frvp_proxy(btc_c, btc_v, btc_start, b2_pos)

            # ── FRVP for ETH ──────────────────────────────────────
            eth_start = last_sl(eth_sl, e1_pos)
            eth_poc, eth_vah = calc_frvp_proxy(eth_c, eth_v, eth_start, e2_pos)

            seen_conf_pos.add(confirmed_pos)
            signals.append({
                "btc_sh1_pos":   b1_pos,
                "btc_sh1_price": round(b1_p, 4),
                "btc_sh2_pos":   b2_pos,
                "btc_sh2_price": round(b2_p, 4),
                "btc_hh":        btc_hh,
                "eth_sh1_pos":   e1_pos,
                "eth_sh1_price": round(e1_p, 4),
                "eth_sh2_pos":   e2_pos,
                "eth_sh2_price": round(e2_p, 4),
                "eth_hh":        eth_hh,
                "confirmed_pos": confirmed_pos,
                "btc_poc":       round(btc_poc, 4),
                "btc_vah":       round(btc_vah, 4),
                "btc_range_start": btc_start,
                "eth_poc":       round(eth_poc, 4),
                "eth_vah":       round(eth_vah, 4),
                "eth_range_start": eth_start,
            })

    signals.sort(key=lambda s: s["confirmed_pos"])
    return signals


# ══════════════════════════════════════════════════════════════════════════
# 6. Backtest engine
# ══════════════════════════════════════════════════════════════════════════

def run_backtest(df_btc: pd.DataFrame, df_eth: pd.DataFrame,
                 signals: list, cfg: dict) -> pd.DataFrame:
    """
    Simulates bearish (short) trades based on confirmed divergence signals.

    State machine:
      IDLE:      no open trade, no active watch
      WATCHING:  first asset broke POC, waiting up to poc_break_bars for second
      IN_TRADE:  both assets shorted, monitoring stop/TP

    PnL: equal-weight average of BTC and ETH.
    """
    fee        = cfg["fee_pct"] / 100
    vah_buf    = cfg["vah_buffer_pct"] / 100
    rsi_thr    = cfg["tp_rsi_threshold"]
    poc_bars   = cfg["poc_break_bars"]
    sig_exp    = cfg["signal_expiry_bars"]

    n          = len(df_btc)
    btc_open   = df_btc["open"].values
    btc_close  = df_btc["close"].values
    btc_high   = df_btc["high"].values
    eth_open   = df_eth["open"].values
    eth_close  = df_eth["close"].values
    eth_high   = df_eth["high"].values
    btc_rsi    = calc_rsi(df_btc["close"], cfg["tp_rsi_period"])

    # Index signals by confirmed_pos for O(1) lookup
    sig_map = defaultdict(list)
    for s in signals:
        sig_map[s["confirmed_pos"]].append(s)

    trades  = []
    pending = []        # confirmed signals not yet triggered

    # WATCHING state
    watching    = False
    watch_first = None  # 'btc' or 'eth'
    watch_bar   = None
    watch_sig   = None

    # IN_TRADE state
    in_pos      = False
    entry_bar   = None
    entry_btc   = None
    entry_eth   = None
    stop_btc    = None
    stop_eth    = None
    rsi_cnt     = 0      # 0=off, 1=first bar below rsi_thr
    exit_next   = False  # flag: exit at OPEN of next bar (TP countdown complete)

    for i in range(1, n):
        # ── New confirmed signals ──────────────────────────────────
        if i in sig_map:
            pending.extend(sig_map[i])

        # ── Expire stale signals ───────────────────────────────────
        pending = [s for s in pending if i - s["confirmed_pos"] <= sig_exp]

        # ══════════════════════════════════════════════════════════
        # NOT IN POSITION
        # ══════════════════════════════════════════════════════════
        if not in_pos:
            if not watching:
                # Check all pending signals for POC break
                for s in pending:
                    b_broke = btc_close[i] < s["btc_poc"]
                    e_broke = eth_close[i] < s["eth_poc"]

                    if b_broke and e_broke:
                        # Both broken same bar → enter next bar
                        _open_trade(trades, i + 1, s, btc_open, eth_open,
                                    fee, vah_buf, df_btc,
                                    btc_high=btc_high, eth_high=eth_high,
                                    stop_mode=cfg["stop_mode"])
                        in_pos = entry_bar = i + 1
                        in_pos     = True
                        entry_bar  = i + 1
                        entry_btc  = trades[-1]["entry_btc"]
                        entry_eth  = trades[-1]["entry_eth"]
                        stop_btc   = trades[-1]["stop_btc"]
                        stop_eth   = trades[-1]["stop_eth"]
                        rsi_cnt    = 0
                        exit_next  = False
                        pending    = []
                        break

                    elif b_broke:
                        watching    = True
                        watch_first = "btc"
                        watch_bar   = i
                        watch_sig   = s
                        break

                    elif e_broke:
                        watching    = True
                        watch_first = "eth"
                        watch_bar   = i
                        watch_sig   = s
                        break

            else:
                # Waiting for second asset to confirm
                if i - watch_bar > poc_bars:
                    # Timeout — cancel watch
                    watching    = False
                    watch_first = None
                    watch_bar   = None
                    watch_sig   = None
                else:
                    confirmed = False
                    if watch_first == "btc" and eth_close[i] < watch_sig["eth_poc"]:
                        confirmed = True
                    elif watch_first == "eth" and btc_close[i] < watch_sig["btc_poc"]:
                        confirmed = True

                    if confirmed and i + 1 < n:
                        _open_trade(trades, i + 1, watch_sig, btc_open, eth_open,
                                    fee, vah_buf, df_btc,
                                    btc_high=btc_high, eth_high=eth_high,
                                    stop_mode=cfg["stop_mode"])
                        in_pos     = True
                        entry_bar  = i + 1
                        entry_btc  = trades[-1]["entry_btc"]
                        entry_eth  = trades[-1]["entry_eth"]
                        stop_btc   = trades[-1]["stop_btc"]
                        stop_eth   = trades[-1]["stop_eth"]
                        rsi_cnt    = 0
                        exit_next  = False
                        pending    = []
                        watching   = False

        # ══════════════════════════════════════════════════════════
        # IN POSITION
        # ══════════════════════════════════════════════════════════
        else:
            # We entered at entry_bar; first possible exit check is at entry_bar + 1.
            # But we track entry_bar as the bar we entered, so at bar i = entry_bar
            # we are already "in" — skip exit logic for the entry bar itself.
            if i == entry_bar:
                continue

            reason   = None
            exit_b   = None
            exit_e   = None

            # ── Priority 1: TP at open (flagged by previous bar's RSI) ──
            if exit_next:
                reason = "take_profit"
                exit_b = btc_open[i] * (1 + fee)
                exit_e = eth_open[i] * (1 + fee)

            # ── Priority 2: Stop loss (intrabar high reaches stop) ───────
            elif btc_high[i] >= stop_btc or eth_high[i] >= stop_eth:
                reason = "stop_loss"
                exit_b = max(stop_btc, btc_close[i]) * (1 + fee)
                exit_e = max(stop_eth, eth_close[i]) * (1 + fee)

            if reason:
                _close_trade(trades, i, df_btc, exit_b, exit_e, reason, entry_bar)
                in_pos    = False
                exit_next = False
                rsi_cnt   = 0
                pending   = []

            else:
                # ── Update RSI countdown (at bar i close) ───────────────
                if rsi_cnt == 1:
                    if btc_rsi[i] >= rsi_thr:
                        rsi_cnt = 0       # RSI bounced back above 30 → reset
                    else:
                        exit_next = True  # 2nd consecutive bar < 30 → exit at next open
                else:
                    if btc_rsi[i] < rsi_thr:
                        rsi_cnt = 1       # first bar below 30

    return pd.DataFrame(trades)


def _open_trade(trades, bar_idx, sig, btc_open, eth_open,
                fee, vah_buf, df_btc,
                btc_high=None, eth_high=None, stop_mode="prev_high"):
    """Append a new trade entry record."""
    e_btc = btc_open[bar_idx] * (1 - fee)   # short sell: fee reduces effective sell
    e_eth = eth_open[bar_idx] * (1 - fee)

    prev = bar_idx - 1  # bar just before entry (= POC-break confirmation bar)
    if stop_mode == "prev_high" and btc_high is not None:
        s_btc = btc_high[prev] * (1 + vah_buf)
        s_eth = eth_high[prev] * (1 + vah_buf)
    elif stop_mode == "sh2":
        s_btc = sig["btc_sh2_price"] * (1 + vah_buf)
        s_eth = sig["eth_sh2_price"] * (1 + vah_buf)
    else:  # "vah" (original)
        s_btc = sig["btc_vah"] * (1 + vah_buf)
        s_eth = sig["eth_vah"] * (1 + vah_buf)

    trades.append({
        # Entry
        "entry_idx":      bar_idx,
        "entry_time":     df_btc.index[bar_idx],
        "entry_btc":      round(e_btc, 4),
        "entry_eth":      round(e_eth, 4),
        # Risk levels
        "stop_btc":       round(s_btc, 4),
        "stop_eth":       round(s_eth, 4),
        "btc_poc":        sig["btc_poc"],
        "eth_poc":        sig["eth_poc"],
        "btc_vah":        sig["btc_vah"],
        "eth_vah":        sig["eth_vah"],
        "sl_dist_btc":    round(abs(s_btc - e_btc) / e_btc * 100, 3),
        "sl_dist_eth":    round(abs(s_eth - e_eth) / e_eth * 100, 3),
        # Signal context
        "div_type":       "BTC_HH+ETH_LH" if sig["btc_hh"] else "BTC_LH+ETH_HH",
        "btc_sh2_price":  sig["btc_sh2_price"],
        "eth_sh2_price":  sig["eth_sh2_price"],
        "confirmed_pos":  sig["confirmed_pos"],
    })


def _close_trade(trades, bar_idx, df_btc, exit_b, exit_e, reason, entry_bar):
    """Update the last trade record with exit info."""
    t = trades[-1]
    pnl_b = (t["entry_btc"] - exit_b) / t["entry_btc"] * 100
    pnl_e = (t["entry_eth"] - exit_e) / t["entry_eth"] * 100
    pnl   = (pnl_b + pnl_e) / 2

    t.update({
        "exit_idx":    bar_idx,
        "exit_time":   df_btc.index[bar_idx],
        "exit_btc":    round(exit_b, 4),
        "exit_eth":    round(exit_e, 4),
        "pnl_btc":     round(pnl_b, 4),
        "pnl_eth":     round(pnl_e, 4),
        "pnl_avg":     round(pnl, 4),
        "win":         pnl > 0,
        "exit_reason": reason,
        "bars_held":   bar_idx - entry_bar,
        "year":        df_btc.index[bar_idx].year,
    })


# ══════════════════════════════════════════════════════════════════════════
# 7. Report
# ══════════════════════════════════════════════════════════════════════════

def report(df: pd.DataFrame, df_btc: pd.DataFrame, cfg: dict,
           n_signals: int = 0) -> str:
    DIV  = "═" * 65
    div  = "─" * 65
    lines = []

    def hdr(title):
        lines.append(f"\n{DIV}")
        lines.append(f"  {title}")
        lines.append(DIV)

    def metrics(sub, label):
        if sub.empty:
            lines.append(f"  {label}: （無交易）")
            return
        comp = sub.dropna(subset=["exit_time"])
        if comp.empty:
            lines.append(f"  {label}: （無完成交易）")
            return
        n_tot = len(comp)
        wins  = comp[comp["win"]]
        loss  = comp[~comp["win"]]
        wr    = len(wins) / n_tot * 100
        aw    = wins["pnl_avg"].mean()  if not wins.empty  else 0.0
        al    = loss["pnl_avg"].mean()  if not loss.empty  else 0.0
        exp   = wr / 100 * aw + (1 - wr / 100) * al
        rr    = abs(aw / al) if al != 0 else float("inf")
        pf    = wins["pnl_avg"].sum() / abs(loss["pnl_avg"].sum()) \
                if not loss.empty and loss["pnl_avg"].sum() != 0 else float("inf")
        lines.append(f"\n  ── {label}")
        lines.append(f"  交易次數   {n_tot}  ({len(wins)} 勝 / {len(loss)} 敗)")
        lines.append(f"  勝率       {wr:.1f}%")
        lines.append(f"  平均盈利   +{aw:.2f}%")
        lines.append(f"  平均虧損   {al:.2f}%")
        lines.append(f"  盈虧比     {rr:.2f}x")
        lines.append(f"  期望值     {exp:+.3f}%")
        lines.append(f"  獲利因子   {pf:.2f}")
        avg_bars = comp["bars_held"].mean()
        lines.append(f"  平均持倉   {avg_bars:.1f} 根棒 ({avg_bars * 4:.0f} 小時)")
        exits = comp["exit_reason"].value_counts().to_dict()
        lines.append(
            f"  出場分佈   止盈={exits.get('take_profit',0)}  "
            f"止損={exits.get('stop_loss',0)}"
        )
        return comp

    hdr("FRVP Cross-Asset Divergence Backtest  ｜  空頭策略")
    lines.append(f"\n  回測期間   {df_btc.index[0].date()} → {df_btc.index[-1].date()}")
    lines.append(f"  K棒總數    {len(df_btc):,} 根 4H")
    lines.append(f"  策略版本   FRVP-VWAP + RSI<{cfg['tp_rsi_threshold']} TP")
    open_trades = df[df["exit_time"].isna()]
    closed      = df.dropna(subset=["exit_time"])
    lines.append(f"  偵測訊號   {n_signals} 個  →  進場 {len(df)} 筆（完成 {len(closed)}，未平倉 {len(open_trades)}）")

    comp_all = metrics(df, "整體績效")

    # By divergence type
    if comp_all is not None and not comp_all.empty:
        hdr("按背離類型分拆")
        for dtype in df["div_type"].unique():
            metrics(df[df["div_type"] == dtype], dtype)

    # By year
    comp = df.dropna(subset=["exit_time"])
    if not comp.empty:
        hdr("按年份分拆")
        lines.append(f"\n  {'年份':<6} {'筆數':>4} {'勝率':>7} {'均盈':>8} {'均損':>8} {'期望值':>9}  市況")
        lines.append(f"  {'─'*60}")
        market = {2021:"牛市", 2022:"熊市", 2023:"復甦", 2024:"牛市", 2025:"牛市?", 2026:"未定"}
        for yr in sorted(comp["year"].unique()):
            yt = comp[comp["year"] == yr]
            yw = yt[yt["win"]]
            yl = yt[~yt["win"]]
            y_wr  = len(yw) / len(yt) * 100
            y_aw  = yw["pnl_avg"].mean() if not yw.empty else 0
            y_al  = yl["pnl_avg"].mean() if not yl.empty else 0
            y_exp = y_wr / 100 * y_aw + (1 - y_wr / 100) * y_al
            ml = market.get(yr, "")
            lines.append(
                f"  {yr:<6} {len(yt):>4} {y_wr:>6.1f}% {y_aw:>+7.2f}% {y_al:>+7.2f}% {y_exp:>+8.3f}%  {ml}"
            )

    # Individual trades table
    hdr("交易明細（依時間）")
    show_cols = ["entry_time", "exit_time", "div_type",
                 "entry_btc", "entry_eth", "pnl_btc", "pnl_eth", "pnl_avg",
                 "exit_reason", "bars_held"]
    comp_full = df.dropna(subset=["exit_time"])
    if not comp_full.empty:
        lines.append("")
        hdr_row = (f"  {'#':>3}  {'進場時間':<17} {'出場時間':<17} "
                   f"{'類型':<20} {'BTC%':>7} {'ETH%':>7} {'均%':>7} "
                   f"{'原因':<12} {'棒數':>5}")
        lines.append(hdr_row)
        lines.append(f"  {'─'*100}")
        for row_i, (_, row) in enumerate(comp_full.iterrows(), 1):
            e_t  = str(row["entry_time"])[:16]
            x_t  = str(row["exit_time"])[:16]
            win_mark = "✓" if row["win"] else "✗"
            lines.append(
                f"  {row_i:>3}  {e_t:<17} {x_t:<17} "
                f"{row['div_type']:<20} "
                f"{row['pnl_btc']:>+7.2f} {row['pnl_eth']:>+7.2f} "
                f"{row['pnl_avg']:>+7.2f} {win_mark} "
                f"{row['exit_reason']:<12} {row['bars_held']:>5}"
            )

    hdr("Key FRVP Levels (每筆進場）")
    if not comp_full.empty:
        lines.append(f"\n  {'#':>3}  {'進場時間':<17}  "
                     f"{'BTC_POC':>9} {'BTC_VAH':>9} {'BTC_Stop':>9}  "
                     f"{'ETH_POC':>9} {'ETH_VAH':>9} {'ETH_Stop':>9}")
        lines.append(f"  {'─'*90}")
        for row_i, (_, row) in enumerate(comp_full.iterrows(), 1):
            lines.append(
                f"  {row_i:>3}  {str(row['entry_time'])[:16]:<17}  "
                f"{row['btc_poc']:>9.1f} {row['btc_vah']:>9.1f} {row['stop_btc']:>9.1f}  "
                f"{row['eth_poc']:>9.2f} {row['eth_vah']:>9.2f} {row['stop_eth']:>9.2f}"
            )

    hdr("風險提示")
    lines.append("  1. 以上為歷史回測，未來績效不保證重現")
    lines.append("  2. 已含 0.06% 手續費（每邊），未含大額滑點影響")
    lines.append(f"  3. FRVP POC = VWAP 代理（非真實成交量分佈）")
    lines.append(f"  4. VAH = 85th 百分位收盤（近似值）")
    lines.append(f"  5. Swing High 確認需 {cfg['swing_window']} 根後續棒（天然延遲）")
    lines.append(f"  6. 止損模式：{cfg.get('stop_mode','vah')}  buffer={cfg['vah_buffer_pct']}%")
    lines.append(f"  7. 止盈：BTC RSI 連續 2 根 4H 收盤 < {cfg['tp_rsi_threshold']}")
    lines.append(f"\n{DIV}\n")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# 8. CLI & Main
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="FRVP Cross-Asset Divergence Backtest")
    p.add_argument("--swing-window",    type=int,   default=DEFAULT_CFG["swing_window"])
    p.add_argument("--min-gap",         type=int,   default=DEFAULT_CFG["min_div_gap"])
    p.add_argument("--max-gap",         type=int,   default=DEFAULT_CFG["max_div_gap"])
    p.add_argument("--min-pct",         type=float, default=DEFAULT_CFG["min_price_diff_pct"])
    p.add_argument("--sync-tol",        type=int,   default=DEFAULT_CFG["sync_tolerance"])
    p.add_argument("--max-frvp-lb",     type=int,   default=DEFAULT_CFG["max_frvp_lookback"])
    p.add_argument("--signal-expiry",   type=int,   default=DEFAULT_CFG["signal_expiry_bars"])
    p.add_argument("--poc-bars",        type=int,   default=DEFAULT_CFG["poc_break_bars"])
    p.add_argument("--vah-buffer",      type=float, default=DEFAULT_CFG["vah_buffer_pct"])
    p.add_argument("--stop-mode",       type=str,   default=DEFAULT_CFG["stop_mode"],
                   choices=["vah", "prev_high", "sh2"])
    p.add_argument("--rsi-period",      type=int,   default=DEFAULT_CFG["tp_rsi_period"])
    p.add_argument("--rsi-threshold",   type=int,   default=DEFAULT_CFG["tp_rsi_threshold"])
    p.add_argument("--fee",             type=float, default=DEFAULT_CFG["fee_pct"])
    p.add_argument("--start-date",      type=str,   default="2019-01-01")
    p.add_argument("--end-date",        type=str,   default="2025-12-31")
    p.add_argument("--output-dir",      type=str,   default=str(OUTPUT_DIR))
    p.add_argument("--results-dir",     type=str,   default=str(RESULTS_DIR))
    return p.parse_args()


def main():
    args       = parse_args()
    out_dir    = Path(args.output_dir)
    res_dir    = Path(args.results_dir)
    res_dir.mkdir(exist_ok=True)

    cfg = {
        "swing_window":       args.swing_window,
        "min_div_gap":        args.min_gap,
        "max_div_gap":        args.max_gap,
        "min_price_diff_pct": args.min_pct,
        "sync_tolerance":     args.sync_tol,
        "max_frvp_lookback":  args.max_frvp_lb,
        "signal_expiry_bars": args.signal_expiry,
        "poc_break_bars":     args.poc_bars,
        "vah_buffer_pct":     args.vah_buffer,
        "stop_mode":          args.stop_mode,
        "tp_rsi_period":      args.rsi_period,
        "tp_rsi_threshold":   args.rsi_threshold,
        "fee_pct":            args.fee,
    }

    SEP = "═" * 65
    print(SEP)
    print("  FRVP Cross-Asset Divergence Backtest  (frvp_backtest.py)")
    print(SEP)
    print(f"  swing_window={cfg['swing_window']}  "
          f"div_gap=[{cfg['min_div_gap']},{cfg['max_div_gap']}]  "
          f"min_pct={cfg['min_price_diff_pct']}%")
    print(f"  sync_tol={cfg['sync_tolerance']}  "
          f"signal_expiry={cfg['signal_expiry_bars']}  "
          f"poc_break_bars={cfg['poc_break_bars']}")
    print(f"  vah_buffer={cfg['vah_buffer_pct']}%  stop_mode={cfg['stop_mode']}  "
          f"tp_rsi<{cfg['tp_rsi_threshold']}  fee={cfg['fee_pct']}%/side")
    print()

    # ── Fetch data from Binance (same source as MACD backtest) ────────
    print("📡 Fetching OHLCV from Binance...")
    df_btc = fetch_binance_ohlcv("BTCUSDT", "4h", args.start_date, args.end_date)
    df_eth = fetch_binance_ohlcv("ETHUSDT", "4h", args.start_date, args.end_date)

    # ── Align on common timestamps ─────────────────────────────────
    common = df_btc.index.intersection(df_eth.index)
    df_btc = df_btc.loc[common].copy()
    df_eth = df_eth.loc[common].copy()
    print(f"  Aligned: {len(df_btc):,} candles  "
          f"({df_btc.index[0].date()} → {df_btc.index[-1].date()})")

    # ── Detect divergences ─────────────────────────────────────────
    print("\n🔍 Detecting FRVP divergences...")
    signals = detect_divergences(df_btc, df_eth, cfg)
    print(f"   → {len(signals)} divergence signals found")
    if signals:
        print(f"   → BTC_HH+ETH_LH : "
              f"{sum(1 for s in signals if s['btc_hh'])}")
        print(f"   → BTC_LH+ETH_HH : "
              f"{sum(1 for s in signals if not s['btc_hh'])}")

    if not signals:
        print("\n⚠️  No signals detected. Try loosening parameters.")
        sys.exit(0)

    # ── Run backtest ───────────────────────────────────────────────
    print("\n🚀 Running backtest...")
    trades = run_backtest(df_btc, df_eth, signals, cfg)
    comp   = trades.dropna(subset=["exit_time"])
    print(f"   → {len(comp)} completed trades")

    # ── Report ─────────────────────────────────────────────────────
    rpt = report(trades, df_btc, cfg, n_signals=len(signals))
    print(rpt)

    # ── Save outputs ───────────────────────────────────────────────
    trades_path = res_dir / "frvp_trades.csv"
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    print(f"💾 Trade log  → {trades_path}")

    rpt_path = res_dir / "frvp_report.txt"
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(f"Config: {cfg}\n\n")
        f.write(rpt)
    print(f"💾 Report     → {rpt_path}")


if __name__ == "__main__":
    main()
