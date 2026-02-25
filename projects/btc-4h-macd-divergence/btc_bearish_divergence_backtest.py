#!/usr/bin/env python3
"""
BTC MACD 頂背離策略 回測系統（空頭版）
=================================
策略規則：
  進場（做空）：MACD 4H 頂背離確認 + 頂部回落 2% + RSI 60~70
  止盈：4H RSI <= 30
  止損：頂背離高點（略上方 0.1% buffer）
  無時間止損
"""
import os
import numpy as np
import pandas as pd
import requests
import time
import warnings
from datetime import datetime, timezone

warnings.filterwarnings('ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results")


# ================================================================
# 1. 資料抓取（Binance 免費 API）
# ================================================================

def fetch_binance_ohlcv(symbol: str, interval: str,
                         start_date: str, end_date: str) -> pd.DataFrame:
    print(f"📡 Fetching {symbol} {interval} from {start_date} to {end_date}...")
    url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d")
                 .replace(tzinfo=timezone.utc).timestamp() * 1000)
    all_klines = []
    current_ts = start_ts
    while current_ts < end_ts:
        params = {'symbol': symbol, 'interval': interval,
                  'startTime': current_ts, 'endTime': end_ts, 'limit': 1000}
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            klines = resp.json()
        except Exception as e:
            print(f"⚠️  API error: {e}, retrying...")
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
        raise ValueError("No data fetched!")
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_vol', 'trades', 'tb_base', 'tb_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df = df[~df.index.duplicated(keep='first')].sort_index()
    print(f"✅ {len(df):,} candles  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ================================================================
# 2. 技術指標
# ================================================================

def calc_ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def calc_macd(close, fast=12, slow=26, signal=9):
    macd_line = calc_ema(close, fast) - calc_ema(close, slow)
    signal_line = calc_ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def calc_rsi(close, period=14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    ag = gain.ewm(com=period - 1, adjust=False).mean()
    al = loss.ewm(com=period - 1, adjust=False).mean()
    rs = ag / (al + 1e-10)
    return 100 - (100 / (1 + rs))


# ================================================================
# 3. Swing High 偵測（無未來資訊）
# ================================================================

def find_swing_highs(price: pd.Series, window: int = 5) -> pd.Series:
    sh = pd.Series(False, index=price.index)
    arr = price.values
    n = len(arr)
    for i in range(window, n - window):
        local_max = arr[i - window: i + window + 1].max()
        if arr[i] == local_max and arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            sh.iloc[i] = True
    return sh


# ================================================================
# 4. MACD 頂背離偵測（正確定義版）
# ================================================================

def detect_bearish_divergences(df: pd.DataFrame,
                                swing_window: int = 5,
                                max_high1_lookback: int = 200,
                                debug: bool = False) -> pd.DataFrame:
    """
    MACD 頂背離序列（偵測兩種）：
      High #1  → MACD 雙線穿越零軸以下（離開 histogram 正區）→ 價格觸碰 EMA52（從上方）
              → High #2

    Regular Bearish：Price Higher High + MACD Lower High（趨勢反轉）
    Hidden  Bearish：Price Lower High + MACD Higher High（熊市反彈延續）
    """
    close    = df['close'].values
    macd_l   = df['macd_line'].values
    signal_l = df['signal_line'].values
    ema52    = df['ema52'].values
    high_arr = df['high'].values
    n = len(df)

    swing_highs_mask = find_swing_highs(df['close'], swing_window)
    swing_high_positions = [i for i in range(n) if swing_highs_mask.iloc[i]]

    df = df.copy()
    df['div_signal']        = False
    df['div_high_price']    = np.nan
    df['div_type']          = ''
    df['div_confirmed_pos'] = -1

    high1_pos = None

    for sh_pos in swing_high_positions:
        # 只看 MACD 在正值區的 swing high
        if macd_l[sh_pos] <= 0:
            continue

        sh_date = df.index[sh_pos]
        in_debug_range = debug and str(sh_date)[:4] >= '2021'

        if high1_pos is None:
            if in_debug_range:
                print(f"  [DBG] {sh_date} pos={sh_pos}  close={close[sh_pos]:.0f}  macd={macd_l[sh_pos]:.1f}  → SET as High1 (was None)")
            high1_pos = sh_pos
            continue

        between = range(high1_pos + 1, sh_pos)

        # 條件 A：MACD 線與 Signal 線雙雙穿越零軸以下（離開正 histogram）
        macd_recovered = any(
            macd_l[j] < 0 and signal_l[j] < 0
            for j in between
        )

        # 條件 B：價格觸碰 EMA52（從上方回落）
        ema52_touched = any(close[j] <= ema52[j] for j in between)

        if in_debug_range:
            high1_date = df.index[high1_pos]
            print(f"  [DBG] {sh_date} pos={sh_pos}  close={close[sh_pos]:.0f}  macd={macd_l[sh_pos]:.1f}")
            print(f"         vs High1 @ {high1_date}  close={close[high1_pos]:.0f}  macd={macd_l[high1_pos]:.1f}")
            print(f"         recovery: macd_ok={macd_recovered}  ema52_ok={ema52_touched}")

        if macd_recovered and ema52_touched:
            price_higher_high = close[sh_pos] > close[high1_pos]
            macd_lower_high   = macd_l[sh_pos] < macd_l[high1_pos]
            price_lower_high  = close[sh_pos] < close[high1_pos]
            macd_higher_high  = macd_l[sh_pos] > macd_l[high1_pos]

            regular_div = price_higher_high and macd_lower_high  # 標準頂背離
            hidden_div  = price_lower_high and macd_higher_high  # 隱性頂背離

            if in_debug_range:
                print(f"         → regular={regular_div} hidden={hidden_div}")

            if regular_div or hidden_div:
                confirmed_pos = min(sh_pos + swing_window, n - 1)
                div_type = 'regular' if regular_div else 'hidden'
                # div_high 取確認窗口內的最高價
                window_end = confirmed_pos + 1
                div_high_price = float(np.max(high_arr[sh_pos:window_end]))
                df.iloc[sh_pos, df.columns.get_loc('div_signal')]        = True
                df.iloc[sh_pos, df.columns.get_loc('div_high_price')]    = div_high_price
                df.iloc[sh_pos, df.columns.get_loc('div_type')]          = div_type
                df.iloc[sh_pos, df.columns.get_loc('div_confirmed_pos')] = confirmed_pos
                if in_debug_range:
                    print(f"         ✅ DIV DETECTED type={div_type}  div_high={div_high_price:.0f}")
                high1_pos = sh_pos
            else:
                # 完整恢復後沒有背離 → 決定是否更新 High1
                # (A) 新 swing high 價格 > 舊 High1 → 更新（更高的錨點）
                # (B) 舊 High1 距今超過 max_high1_lookback 棒 → 更新（zombie anchor）
                # (C) 否則保持舊 High1（防止較低的 swing high 覆蓋更有意義的 High1）
                price_higher = close[sh_pos] > close[high1_pos]
                high1_stale = (sh_pos - high1_pos) > max_high1_lookback
                if price_higher or high1_stale:
                    if in_debug_range:
                        reason = 'price higher' if price_higher else 'stale High1'
                        print(f"         → recovery but no div ({reason}) → UPDATE High1")
                    high1_pos = sh_pos
                else:
                    if in_debug_range:
                        print(f"         → recovery but no div, price lower → KEEP High1")
        else:
            # 尚未完整恢復 → 保留價格更高者作為 High #1 參考點
            if close[sh_pos] >= close[high1_pos]:
                if in_debug_range:
                    print(f"         → no recovery, price higher/equal → UPDATE High1")
                high1_pos = sh_pos
            else:
                if in_debug_range:
                    print(f"         → no recovery, price lower → KEEP High1")

    return df


# ================================================================
# 5. 回測引擎（空頭）
# ================================================================

def run_backtest(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rebound_pct  = cfg['rebound_pct']   # price drops this % from div_high → entry
    rsi_min      = cfg['rsi_min']       # RSI 入場範圍下限（空頭：60）
    rsi_max      = cfg['rsi_max']       # RSI 入場範圍上限（空頭：70）
    tp_rsi       = cfg['tp_rsi']        # 止盈 RSI 觸發值（空頭：≤30）
    div_expiry   = cfg['div_expiry_bars']
    fee          = cfg['fee_pct'] / 100

    close        = df['close'].values
    high         = df['high'].values
    low          = df['low'].values
    rsi          = df['rsi'].values
    div_signal   = df['div_signal'].values
    div_high     = df['div_high_price'].values
    div_type_arr = df['div_type'].values
    div_conf_pos = df['div_confirmed_pos'].values

    conf_map = {}
    for i in range(len(df)):
        if div_signal[i]:
            cp = int(div_conf_pos[i])
            if cp >= 0:
                if cp not in conf_map:
                    conf_map[cp] = []
                conf_map[cp].append((cp, div_high[i], div_type_arr[i]))

    trades   = []
    in_pos   = False
    entry_price = stop_loss = entry_bar = None
    pending_divs = []   # (confirmed_at_bar, div_high_price, div_type)

    for i in range(1, len(df)):
        if i in conf_map:
            pending_divs.extend(conf_map[i])

        if not in_pos:
            # 移除過期的背離信號
            pending_divs = [
                (conf_at, d_high, d_type) for (conf_at, d_high, d_type) in pending_divs
                if i - conf_at <= div_expiry
            ]
            for (conf_at, d_high, d_type) in pending_divs[:]:
                threshold = d_high * (1 - rebound_pct / 100)  # price drops 2%
                # RSI 在 60-70 範圍（允許前一棒剛從 >70 下來）
                rsi_ok = (rsi_min <= rsi[i] <= rsi_max) or (rsi_min <= rsi[i - 1] <= rsi_max)
                if close[i] <= threshold and rsi_ok:
                    in_pos      = True
                    entry_price = close[i] * (1 - fee)   # 做空：賣出，扣手續費
                    stop_loss   = d_high * (1 + 0.001)   # 止損在頂背離高點上方
                    entry_bar   = i
                    trades.append({
                        'entry_idx':   i,
                        'entry_time':  df.index[i],
                        'entry_price': entry_price,
                        'stop_loss':   stop_loss,
                        'div_high':    d_high,
                        'div_type':    d_type,
                        'entry_rsi':   rsi[i],
                        'sl_pct':      (stop_loss - entry_price) / entry_price * 100,
                    })
                    pending_divs = []
                    break
        else:
            bars    = i - entry_bar
            reason  = None
            exit_px = None

            # 止損：價格突破頂背離高點（空頭被軋空）
            if high[i] >= stop_loss:
                reason  = 'stop_loss'
                exit_px = max(stop_loss, close[i]) * (1 + fee)
            # 止盈：RSI ≤ 30（超賣）
            elif rsi[i] <= tp_rsi:
                reason  = 'take_profit'
                exit_px = close[i] * (1 + fee)

            if reason:
                in_pos = False
                # 空頭盈虧：賣出 - 回補 / 賣出
                pnl = (entry_price - exit_px) / entry_price * 100
                trades[-1].update({
                    'exit_idx':    i,
                    'exit_time':   df.index[i],
                    'exit_price':  exit_px,
                    'exit_reason': reason,
                    'bars_held':   bars,
                    'pnl_pct':     pnl,
                    'win':         pnl > 0,
                    'year':        df.index[i].year,
                })
                pending_divs = []

    return pd.DataFrame(trades)


# ================================================================
# 6. 報告輸出
# ================================================================

def report_md(trades_df: pd.DataFrame, df: pd.DataFrame, cfg: dict) -> str:
    comp = trades_df.dropna(subset=['exit_time']).copy() if not trades_df.empty else pd.DataFrame()
    if comp.empty:
        return "# 回測報告\n\n❌ 無完成交易。\n"

    total    = len(comp)
    wins     = comp[comp['win'] == True]
    losses   = comp[comp['win'] == False]
    wr       = len(wins) / total * 100
    avg_win  = wins['pnl_pct'].mean()   if len(wins)   > 0 else 0.0
    avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0.0
    expectancy = (wr / 100 * avg_win) + ((1 - wr / 100) * avg_loss)
    pf  = wins['pnl_pct'].sum() / abs(losses['pnl_pct'].sum()) \
          if losses['pnl_pct'].sum() != 0 else float('inf')
    rr  = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    avg_bars = comp['bars_held'].mean()

    exit_rows = []
    for reason, label in [('take_profit', '止盈 RSI≤30'),
                           ('stop_loss',   '止損 頂背離高點')]:
        sub = comp[comp['exit_reason'] == reason]
        if len(sub) == 0:
            continue
        exit_rows.append(f"| {label} | {len(sub)} | {len(sub)/total*100:.1f}% | {sub['pnl_pct'].mean():+.2f}% |")

    market_labels = {2019:'震盪→牛',2020:'牛市',2021:'強牛→熊轉',
                     2022:'熊市',2023:'復甦',2024:'牛市',2025:'牛市/未定'}
    year_rows = []
    for yr in sorted(comp['year'].unique()):
        yt = comp[comp['year'] == yr]
        yw = yt[yt['win'] == True]
        yl = yt[yt['win'] == False]
        y_wr = len(yw) / len(yt) * 100
        y_aw = yw['pnl_pct'].mean() if len(yw) > 0 else 0.0
        y_al = yl['pnl_pct'].mean() if len(yl) > 0 else 0.0
        y_ex = (y_wr / 100 * y_aw) + ((1 - y_wr / 100) * y_al)
        ml   = market_labels.get(yr, '')
        year_rows.append(
            f"| {yr} | {ml} | {len(yt)} | {y_wr:.1f}% "
            f"| {y_aw:+.2f}% | {y_al:+.2f}% | {y_ex:+.3f}% |"
        )

    trade_rows = []
    for _, r in comp.sort_values('entry_time').iterrows():
        icon = '✅' if r['win'] else '❌'
        reason_map = {'take_profit':'止盈','stop_loss':'止損'}
        entry_date = str(r['entry_time'])[:10]
        exit_date  = str(r['exit_time'])[:10]
        trade_rows.append(
            f"| {icon} | {entry_date} | {r['entry_price']:,.0f} | "
            f"{r['stop_loss']:,.0f} | {r['sl_pct']:.1f}% | "
            f"{r.get('div_type','')[:3]} | {r['entry_rsi']:.1f} | "
            f"{exit_date} | {r['exit_price']:,.0f} | "
            f"{reason_map.get(r['exit_reason'], r['exit_reason'])} | "
            f"{r['bars_held']} | {r['pnl_pct']:+.2f}% |"
        )

    run_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    symbol   = cfg.get('symbol', 'BTCUSDT')
    interval = cfg.get('interval', '4h')
    start    = cfg.get('start_date', '')
    end      = cfg.get('end_date', '')

    md = f"""# {symbol} {interval} MACD 頂背離策略（空頭）｜ 回測報告

> 生成時間：{run_date}
> 回測期間：{start} → {end}（共 {len(df):,} 根 {interval} K棒）
> 策略版本：EMA52 回踩 + MACD 穿越零軸 + RSI 60-70 + 2% 回落，偵測 Regular & Hidden 頂背離（做空）

---

## 核心績效

| 指標 | 數值 |
|------|------|
| 完成交易筆數 | {total} 筆 |
| 勝率 | **{wr:.1f}%** （{len(wins)} 勝 / {len(losses)} 敗） |
| 平均盈利 | +{avg_win:.2f}% |
| 平均虧損 | {avg_loss:.2f}% |
| 盈虧比 | {rr:.2f}x |
| 每筆期望值 | **{expectancy:+.3f}%** |
| 獲利因子 | {pf:.2f} |
| 平均持倉 | {avg_bars:.1f} 棒（{avg_bars*4:.0f} 小時） |

---

## 出場原因分布

| 出場方式 | 筆數 | 佔比 | 均報酬 |
|----------|------|------|--------|
""" + "\n".join(exit_rows) + f"""

---

## 按年份分拆

| 年份 | 市況 | 筆數 | 勝率 | 均盈 | 均損 | 期望值 |
|------|------|------|------|------|------|--------|
""" + "\n".join(year_rows) + f"""

---

## 完整交易明細

| | 進場日 | 進場價 | 止損價 | 止損% | 類型 | RSI | 出場日 | 出場價 | 出場方式 | 持棒 | 報酬% |
|-|--------|--------|--------|-------|------|-----|--------|--------|----------|------|-------|
""" + "\n".join(trade_rows) + f"""

---

## 策略參數

| 參數 | 值 |
|----|-----|
| 標的 | {symbol} |
| 週期 | {interval} |
| 方向 | 空頭（做空） |
| MACD | ({cfg.get('macd_fast',12)}, {cfg.get('macd_slow',26)}, {cfg.get('macd_signal',9)}) |
| EMA52 | {cfg.get('ema52_period',52)} |
| RSI 進場範圍 | {cfg.get('rsi_min',60)} ~ {cfg.get('rsi_max',70)} |
| 回落確認 | {cfg.get('rebound_pct',2.0)}% |
| 止盈 RSI | ≤ {cfg.get('tp_rsi',30)} |
| 信號有效期 | {cfg.get('div_expiry_bars',60)} 棒 |
| High1 最大追溯 | {cfg.get('max_high1_lookback',200)} 棒 |
| 手續費 | {cfg.get('fee_pct',0.06)}% 單邊 |

---

## 風險提示

1. 以上為歷史回測，未來績效不保證重現。
2. 已含 {cfg.get('fee_pct',0.06)}% 手續費，未含大額滑點影響。
3. 頂背離偵測延遲 {cfg.get('swing_window',5)} 根棒確認（防未來洩漏）。
4. 本回測未含槓桿；實倉槓桿會放大所有虧損。
5. 策略通過回測不代表可以上實盤，請先做 Paper Trading。
"""
    return md


def report(trades_df: pd.DataFrame, df: pd.DataFrame) -> None:
    if trades_df.empty:
        print("❌ 沒有任何交易記錄")
        return
    comp = trades_df.dropna(subset=['exit_time']).copy()
    if comp.empty:
        print("❌ 沒有完成的交易")
        return
    total = len(comp)
    wins   = comp[comp['win'] == True]
    losses = comp[comp['win'] == False]
    wr     = len(wins) / total * 100
    avg_win  = wins['pnl_pct'].mean()   if len(wins)   > 0 else 0
    avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
    expectancy = (wr / 100 * avg_win) + ((1 - wr / 100) * avg_loss)
    pf = wins['pnl_pct'].sum() / abs(losses['pnl_pct'].sum()) \
        if losses['pnl_pct'].sum() != 0 else float('inf')
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    DIV = "═" * 62
    div = "─" * 62
    print(f"\n{DIV}")
    print("    📊  BTC MACD 頂背離策略（空頭）｜ 歷史回測報告")
    print(DIV)
    print(f"\n  📅 回測期間   {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  🕯️  K棒總數   {len(df):,} 根 4H K棒")
    print(f"  🔔 完成交易   {total} 筆")
    print(f"\n  {div}")
    print(f"  ─── 核心績效")
    print(f"  {div}")
    print(f"  🏆 勝率         {wr:.1f}%   ({len(wins)} 勝 / {len(losses)} 敗)")
    print(f"  📈 平均盈利     +{avg_win:.2f}%")
    print(f"  📉 平均虧損     {avg_loss:.2f}%")
    print(f"  ⚖️  盈虧比       {rr:.2f}x")
    print(f"  💡 每筆期望值   {expectancy:+.3f}%")
    print(f"  🔥 獲利因子     {pf:.2f}")
    print(f"  ⏱️  平均持倉     {comp['bars_held'].mean():.1f} 根棒 ({comp['bars_held'].mean()*4:.0f} 小時)")
    print(f"\n  {div}")
    print(f"  ─── 出場原因分布")
    print(f"  {div}")
    for reason, label in [('take_profit', '止盈 (RSI≤30)'), ('stop_loss', '止損 (頂背離高點)')]:
        sub = comp[comp['exit_reason'] == reason]
        if len(sub) > 0:
            print(f"  {'🟢' if reason == 'take_profit' else '🔴'} {label:<20} {len(sub)} 筆 ({len(sub)/total*100:.1f}%)  均報酬 {sub['pnl_pct'].mean():+.2f}%")
    print(f"\n  {div}")
    print(f"  ─── 按年份分拆")
    print(f"  {div}")
    market_labels = {2019:'震盪→牛',2020:'牛市',2021:'強牛→熊轉',
                     2022:'熊市',2023:'復甦',2024:'牛市',2025:'牛市/未定'}
    print(f"  {'年份':<10} {'筆數':>6} {'勝率':>8} {'均盈':>10} {'均損':>10} {'期望值':>10} {'市況'}")
    print(f"  {'─'*62}")
    for yr in sorted(comp['year'].unique()):
        yt = comp[comp['year'] == yr]
        yw = yt[yt['win'] == True]
        yl = yt[yt['win'] == False]
        y_wr = len(yw) / len(yt) * 100
        y_aw = yw['pnl_pct'].mean() if len(yw) > 0 else 0
        y_al = yl['pnl_pct'].mean() if len(yl) > 0 else 0
        y_ex = (y_wr / 100 * y_aw) + ((1 - y_wr / 100) * y_al)
        ml   = market_labels.get(yr, '')
        print(f"  {yr:<10} {len(yt):>6} {y_wr:>7.1f}%  {y_aw:>+9.2f}%  {y_al:>+9.2f}%  {y_ex:>+9.3f}%  {ml}")
    print(f"\n  {div}")
    print(f"  ⚠️  風險提示")
    print(f"  {div}")
    print(f"  1. 以上為歷史回測，未來績效不保證重現")
    print(f"  2. 已含 0.06% 手續費，未含大額滑點影響")
    print(f"  3. 頂背離偵測延遲 5 根棒確認（防未來洩漏），信號有效期 60 棒（約 10 天）")
    print(f"  4. 本回測未含槓桿；實倉槓桿會放大所有虧損\n")
    print(DIV + "\n")


# ================================================================
# 7. 主程式
# ================================================================

def main():
    cfg = {
        'symbol':             'BTCUSDT',
        'interval':           '4h',
        'start_date':         '2019-01-01',
        'end_date':           '2025-02-25',
        # 技術指標
        'macd_fast':          12,
        'macd_slow':          26,
        'macd_signal':        9,
        'rsi_period':         14,
        'ema52_period':       52,
        # 偵測參數
        'swing_window':       5,
        'max_high1_lookback': 200,
        'div_expiry_bars':    60,
        # 交易條件
        'rsi_min':            60,   # 空頭進場 RSI 下限（從超買區下來）
        'rsi_max':            70,   # 空頭進場 RSI 上限
        'rebound_pct':        2.0,  # 價格從頂背離高點回落 2% 才入場
        'tp_rsi':             30,   # 止盈：RSI ≤ 30（超賣）
        'fee_pct':            0.06,
    }

    df = fetch_binance_ohlcv(cfg['symbol'], cfg['interval'],
                              cfg['start_date'], cfg['end_date'])
    print("⚙️  計算技術指標...")
    df['macd_line'], df['signal_line'], df['histogram'] = calc_macd(
        df['close'], cfg['macd_fast'], cfg['macd_slow'], cfg['macd_signal'])
    df['rsi']   = calc_rsi(df['close'], cfg['rsi_period'])
    df['ema52'] = calc_ema(df['close'], cfg['ema52_period'])

    print("🔍 偵測 MACD 頂背離（EMA52 回踩 + MACD 穿越零軸）...")
    df = detect_bearish_divergences(df, cfg['swing_window'],
                                     max_high1_lookback=cfg.get('max_high1_lookback', 200))
    n_div = df['div_signal'].sum()
    print(f"   → 找到 {n_div} 個頂背離訊號")

    print("🚀 執行回測（空頭）...")
    trades = run_backtest(df, cfg)
    report(trades, df)

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    out_csv  = os.path.join(_RESULTS_DIR, 'btc_macd_bearish_trades.csv')
    out_xlsx = os.path.join(_RESULTS_DIR, 'btc_macd_bearish_trades.xlsx')
    out_md   = os.path.join(_RESULTS_DIR, 'btc_macd_bearish_report.md')

    trades.to_csv(out_csv, index=False)
    md_content = report_md(trades, df, cfg)
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(md_content)
    try:
        trades.to_excel(out_xlsx, index=False, engine='openpyxl')
        print(f"💾 完整交易記錄已儲存：\n   CSV  → {out_csv}\n   XLSX → {out_xlsx}\n   MD   → {out_md}\n")
    except Exception:
        print(f"💾 完整交易記錄已儲存：\n   CSV → {out_csv}\n   MD  → {out_md}\n"
              f"   （xlsx 未寫入，請安裝 openpyxl：pip install openpyxl）")
    return trades, df


if __name__ == '__main__':
    trades, data = main()
