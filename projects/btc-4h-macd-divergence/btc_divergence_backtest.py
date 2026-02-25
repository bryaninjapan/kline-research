#!/usr/bin/env python3
"""
BTC MACD 底背離策略 回測系統（獨立腳本，本子專案用）
=================================
策略規則：
  進場：MACD 4H 底背離確認 + 底部反彈 2% + RSI 30~40
  止盈：4H RSI >= 70
  止損：底背離低點（略下方 0.1% buffer）
  時間止損：入場後 30 根 4H 棒強制出場
"""
import os
import numpy as np
import pandas as pd
import requests
import time
import warnings
from datetime import datetime, timezone

warnings.filterwarnings('ignore')

# 本子專案 results 目錄（與此腳本同層的 results/）
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
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_ts,
            'endTime': end_ts,
            'limit': 1000
        }
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
        'close_time', 'quote_vol', 'trades', 'tb_base', 'tb_quote', 'ignore'
    ])
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
# 3. Swing Low 偵測（無未來資訊）
# ================================================================

def find_swing_lows(price: pd.Series, window: int = 5) -> pd.Series:
    sl = pd.Series(False, index=price.index)
    arr = price.values
    n = len(arr)
    for i in range(window, n - window):
        local_min = arr[i - window: i + window + 1].min()
        if arr[i] == local_min and arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            sl.iloc[i] = True
    return sl


# ================================================================
# 4. MACD 底背離偵測（正確定義版）
# ================================================================

def detect_bullish_divergences(df: pd.DataFrame,
                                swing_window: int = 5,
                                max_low1_lookback: int = 200,
                                debug: bool = False) -> pd.DataFrame:
    """
    MACD 底背離序列（偵測兩種）：
      Low #1  → MACD 雙線穿越零軸（離開 histogram）→ 價格觸碰 EMA52
              → Low #2

    Regular Bullish：Price Lower Low + MACD Higher Low（趨勢反轉）
    Hidden  Bullish：Price Higher Low + MACD Lower Low（牛市回調延續）

    div_low_price 使用 Low #2 棒的最低價（bar low），而非收盤價，
    以確保止損與 2% threshold 計算更準確。
    """
    close    = df['close'].values
    low_arr  = df['low'].values
    macd_l   = df['macd_line'].values
    signal_l = df['signal_line'].values
    ema52    = df['ema52'].values
    n = len(df)

    swing_lows_mask = find_swing_lows(df['close'], swing_window)
    swing_low_positions = [i for i in range(n) if swing_lows_mask.iloc[i]]

    df = df.copy()
    df['div_signal']        = False
    df['div_low_price']     = np.nan   # 使用 bar low（非 close）
    df['div_type']          = ''       # 'regular' or 'hidden'
    df['div_confirmed_pos'] = -1

    low1_pos = None  # integer positional index of Low #1

    for sl_pos in swing_low_positions:
        # 只看 MACD 在負值區的 swing low
        if macd_l[sl_pos] >= 0:
            continue

        sl_date = df.index[sl_pos]
        in_debug_range = debug and str(sl_date)[:7] >= '2024-10'

        if low1_pos is None:
            if in_debug_range:
                print(f"  [DBG] {sl_date} pos={sl_pos}  close={close[sl_pos]:.0f}  macd={macd_l[sl_pos]:.1f}  → SET as Low1 (was None)")
            low1_pos = sl_pos
            continue

        # --- 檢查 Low #1 → Low #2 之間的恢復條件 ---
        between = range(low1_pos + 1, sl_pos)

        # 條件 A：MACD 線與 Signal 線雙雙穿越零軸（離開 histogram）
        macd_recovered = any(
            macd_l[j] > 0 and signal_l[j] > 0
            for j in between
        )

        # 條件 B：價格觸碰 EMA52
        ema52_touched = any(close[j] >= ema52[j] for j in between)

        if in_debug_range:
            low1_date = df.index[low1_pos]
            print(f"  [DBG] {sl_date} pos={sl_pos}  close={close[sl_pos]:.0f}  macd={macd_l[sl_pos]:.1f}")
            print(f"         vs Low1 @ {low1_date}  close={close[low1_pos]:.0f}  macd={macd_l[low1_pos]:.1f}")
            print(f"         recovery: macd_ok={macd_recovered}  ema52_ok={ema52_touched}")

        if macd_recovered and ema52_touched:
            # 完整恢復 → 檢查 divergence 條件（兩種）
            price_lower_low = close[sl_pos] < close[low1_pos]
            macd_higher_low = macd_l[sl_pos] > macd_l[low1_pos]
            price_higher_low = close[sl_pos] > close[low1_pos]
            macd_lower_low   = macd_l[sl_pos] < macd_l[low1_pos]

            regular_div = price_lower_low and macd_higher_low  # 標準底背離
            hidden_div  = price_higher_low and macd_lower_low  # 隱性底背離

            if in_debug_range:
                print(f"         → regular={regular_div} hidden={hidden_div}")

            if regular_div or hidden_div:
                # ✅ 有效底背離（兩種之一）
                confirmed_pos = min(sl_pos + swing_window, n - 1)
                div_type = 'regular' if regular_div else 'hidden'
                # div_low 取確認窗口內的最低價（sl_pos 到 confirmed_pos 之間）
                # 確認後計算，無未來洩漏；比單棒 low 更能捕捉真實底部
                window_end = confirmed_pos + 1
                div_low_price = float(np.min(low_arr[sl_pos:window_end]))
                df.iloc[sl_pos, df.columns.get_loc('div_signal')]        = True
                df.iloc[sl_pos, df.columns.get_loc('div_low_price')]     = div_low_price
                df.iloc[sl_pos, df.columns.get_loc('div_type')]          = div_type
                df.iloc[sl_pos, df.columns.get_loc('div_confirmed_pos')] = confirmed_pos
                if in_debug_range:
                    print(f"         ✅ DIV DETECTED type={div_type}  div_low={div_low_price:.0f}  Low1 → sl_pos")
                # Low2 成為下一個 Low1（不 reset 為 None），
                # 讓後續背離可以以更低的歷史低點為參考
                low1_pos = sl_pos
            else:
                # 完整恢復後沒有背離 → 決定是否更新 Low1
                # 規則：
                #   (A) 新 swing low 價格 < 舊 Low1 → 更新（更深的錨點）
                #   (B) 舊 Low1 距今超過 max_low1_lookback 棒（zombie anchor）→ 更新
                #   (C) 否則保持舊 Low1
                #       → 防止 "Price HL + MACD HL" 的微小 swing low 覆蓋更有意義的 Low1
                price_lower = close[sl_pos] < close[low1_pos]
                low1_stale = (sl_pos - low1_pos) > max_low1_lookback
                if price_lower or low1_stale:
                    if in_debug_range:
                        reason = 'price lower' if price_lower else 'stale Low1'
                        print(f"         → recovery but no div ({reason}) → UPDATE Low1")
                    low1_pos = sl_pos
                else:
                    if in_debug_range:
                        print(f"         → recovery but no div, price higher+macd higher → KEEP Low1")
        else:
            # 尚未完整恢復 → 保留價格更低者作為 Low #1 參考點
            if close[sl_pos] <= close[low1_pos]:
                if in_debug_range:
                    print(f"         → no recovery, price lower/equal → UPDATE Low1")
                low1_pos = sl_pos
            else:
                if in_debug_range:
                    print(f"         → no recovery, price higher → KEEP Low1")

    return df


# ================================================================
# 5. 回測引擎
# ================================================================

def run_backtest(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rebound_pct  = cfg['rebound_pct']
    rsi_min, rsi_max = cfg['rsi_min'], cfg['rsi_max']
    tp_rsi       = cfg['tp_rsi']
    time_stop    = cfg['time_stop_bars']
    div_expiry   = cfg['div_expiry_bars']   # 背離信號最多等幾根棒
    fee          = cfg['fee_pct'] / 100
    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values
    rsi   = df['rsi'].values
    div_signal   = df['div_signal'].values
    div_low      = df['div_low_price'].values
    div_type_arr = df['div_type'].values
    div_conf_pos = df['div_confirmed_pos'].values
    # conf_map: confirmed_bar → list of (confirmed_bar, div_low_price, div_type) tuples
    conf_map = {}
    for i in range(len(df)):
        if div_signal[i]:
            cp = int(div_conf_pos[i])
            if cp >= 0:
                if cp not in conf_map:
                    conf_map[cp] = []
                conf_map[cp].append((cp, div_low[i], div_type_arr[i]))
    trades = []
    in_pos = False
    entry_price = None
    stop_loss = None
    entry_bar = None
    pending_divs = []   # list of (confirmed_at_bar, div_low_price, div_type)
    for i in range(1, len(df)):
        if i in conf_map:
            pending_divs.extend(conf_map[i])
        if not in_pos:
            # 移除過期的背離信號
            pending_divs = [
                (conf_at, d_low, d_type) for (conf_at, d_low, d_type) in pending_divs
                if i - conf_at <= div_expiry
            ]
            for (conf_at, d_low, d_type) in pending_divs[:]:
                threshold = d_low * (1 + rebound_pct / 100)
                # RSI 在 30-40 範圍：允許「前一棒剛從 <30 恢復」+ 本棒 2% 反彈
                rsi_ok = (rsi_min <= rsi[i] <= rsi_max) or (rsi_min <= rsi[i - 1] <= rsi_max)
                if close[i] >= threshold and rsi_ok:
                    in_pos = True
                    entry_price = close[i] * (1 + fee)
                    stop_loss = d_low * (1 - 0.001)
                    entry_bar = i
                    trades.append({
                        'entry_idx': i,
                        'entry_time': df.index[i],
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'div_low': d_low,
                        'div_type': d_type,
                        'entry_rsi': rsi[i],
                        'sl_pct': (entry_price - stop_loss) / entry_price * 100,
                    })
                    pending_divs = []
                    break
        else:
            bars = i - entry_bar
            reason = None
            exit_px = None
            if low[i] <= stop_loss:
                reason = 'stop_loss'
                exit_px = min(stop_loss, close[i]) * (1 - fee)
            elif rsi[i] >= tp_rsi:
                reason = 'take_profit'
                exit_px = close[i] * (1 - fee)
            if reason:
                in_pos = False
                pnl = (exit_px - entry_price) / entry_price * 100
                trades[-1].update({
                    'exit_idx': i,
                    'exit_time': df.index[i],
                    'exit_price': exit_px,
                    'exit_reason': reason,
                    'bars_held': bars,
                    'pnl_pct': pnl,
                    'win': pnl > 0,
                    'year': df.index[i].year,
                })
                pending_divs = []
    return pd.DataFrame(trades)


# ================================================================
# 6. 報告輸出
# ================================================================

def report_md(trades_df: pd.DataFrame, df: pd.DataFrame, cfg: dict) -> str:
    """生成 Markdown 格式回測報告字串，並寫入 results/ 目錄。"""
    comp = trades_df.dropna(subset=['exit_time']).copy() if not trades_df.empty else pd.DataFrame()
    if comp.empty:
        return "# 回測報告\n\n❌ 無完成交易。\n"

    total  = len(comp)
    wins   = comp[comp['win'] == True]
    losses = comp[comp['win'] == False]
    wr     = len(wins) / total * 100
    avg_win  = wins['pnl_pct'].mean()   if len(wins)   > 0 else 0.0
    avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0.0
    expectancy = (wr / 100 * avg_win) + ((1 - wr / 100) * avg_loss)
    pf  = wins['pnl_pct'].sum() / abs(losses['pnl_pct'].sum()) \
          if losses['pnl_pct'].sum() != 0 else float('inf')
    rr  = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    avg_bars = comp['bars_held'].mean()

    # 出場分布
    exit_rows = []
    for reason, label in [('take_profit', '止盈 RSI≥70'),
                           ('stop_loss',   '止損 底背離低點'),
                           ('time_stop',   '時間止損 30棒')]:
        sub = comp[comp['exit_reason'] == reason]
        if len(sub) == 0:
            continue
        exit_rows.append(f"| {label} | {len(sub)} | {len(sub)/total*100:.1f}% | {sub['pnl_pct'].mean():+.2f}% |")

    # 年份分拆
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

    # 交易明細（按時間排序）
    trade_rows = []
    for _, r in comp.sort_values('entry_time').iterrows():
        icon = '✅' if r['win'] else '❌'
        reason_map = {'take_profit':'止盈','stop_loss':'止損','time_stop':'時間止損'}
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

    md = f"""# {symbol} {interval} MACD 底背離策略 ｜ 回測報告

> 生成時間：{run_date}
> 回測期間：{start} → {end}（共 {len(df):,} 根 {interval} K棒）
> 策略版本：EMA52 回踩 + MACD 穿越零軸 + RSI 30-40 + 2% 反彈，偵測 Regular & Hidden 底背離

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
|------|----|
| 標的 | {symbol} |
| 週期 | {interval} |
| MACD | ({cfg.get('macd_fast',12)}, {cfg.get('macd_slow',26)}, {cfg.get('macd_signal',9)}) |
| EMA52 | {cfg.get('ema52_period',52)} |
| RSI 進場範圍 | {cfg.get('rsi_min',30)} ~ {cfg.get('rsi_max',40)} |
| 反彈確認 | {cfg.get('rebound_pct',2.0)}% |
| 止盈 RSI | ≥ {cfg.get('tp_rsi',70)} |
| 時間止損 | {cfg.get('time_stop_bars',30)} 棒 |
| 信號有效期 | {cfg.get('div_expiry_bars',60)} 棒 |
| Low1 最大追溯 | {cfg.get('max_low1_lookback',200)} 棒 |
| 手續費 | {cfg.get('fee_pct',0.06)}% 單邊 |

---

## 風險提示

1. 以上為歷史回測，未來績效不保證重現。
2. 已含 {cfg.get('fee_pct',0.06)}% 手續費，未含大額滑點影響。
3. 底背離偵測延遲 {cfg.get('swing_window',5)} 根棒確認（防未來洩漏）。
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
    wins = comp[comp['win'] == True]
    losses = comp[comp['win'] == False]
    wr = len(wins) / total * 100
    avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
    expectancy = (wr / 100 * avg_win) + ((1 - wr / 100) * avg_loss)
    pf = wins['pnl_pct'].sum() / abs(losses['pnl_pct'].sum()) \
        if losses['pnl_pct'].sum() != 0 else float('inf')
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    DIV = "═" * 62
    div = "─" * 62
    print(f"\n{DIV}")
    print("    📊  BTC MACD 底背離策略 ｜ 歷史回測報告")
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
    labels = {
        'take_profit': ('🟢', '止盈 (RSI≥70)    '),
        'stop_loss':   ('🔴', '止損 (底背離低點)'),
        'time_stop':   ('🟡', '時間止損 (30棒)  '),
    }
    for reason, (em, lb) in labels.items():
        sub = comp[comp['exit_reason'] == reason]
        if len(sub) == 0:
            continue
        avg_p = sub['pnl_pct'].mean()
        print(f"  {em} {lb}  {len(sub):>3} 筆 ({len(sub)/total*100:.1f}%)  均報酬 {avg_p:+.2f}%")
    print(f"\n  {div}")
    print(f"  ─── 按年份分拆")
    print(f"  {div}")
    print(f"  {'年份':<6} {'筆數':>4} {'勝率':>7} {'均盈':>8} {'均損':>8} {'期望值':>9} {'市況'}")
    print(f"  {'─'*58}")
    market_labels = {
        2019: '震盪→牛',
        2020: '牛市',
        2021: '強牛→熊轉',
        2022: '熊市',
        2023: '復甦',
        2024: '牛市',
        2025: '牛市/未定',
    }
    for yr in sorted(comp['year'].unique()):
        yt = comp[comp['year'] == yr]
        yw = yt[yt['win'] == True]
        yl = yt[yt['win'] == False]
        y_wr = len(yw) / len(yt) * 100
        y_aw = yw['pnl_pct'].mean() if len(yw) > 0 else 0
        y_al = yl['pnl_pct'].mean() if len(yl) > 0 else 0
        y_ex = (y_wr / 100 * y_aw) + ((1 - y_wr / 100) * y_al)
        ml = market_labels.get(yr, '')
        print(f"  {yr:<6} {len(yt):>4} {y_wr:>6.1f}% {y_aw:>+7.2f}% {y_al:>+7.2f}% {y_ex:>+8.3f}%  {ml}")
    print(f"\n  {div}")
    print("  ⚠️  風險提示")
    print(f"  {div}")
    print("  1. 以上為歷史回測，未來績效不保證重現")
    print("  2. 已含 0.06% 手續費，未含大額滑點影響")
    print("  3. 底背離偵測延遲 5 根棒確認（防未來洩漏），信號有效期 60 棒（約 10 天）")
    print("  4. 本回測未含槓桿；實倉槓桿會放大所有虧損")
    print(f"\n{DIV}\n")


# ================================================================
# 7. 主程式
# ================================================================

def main():
    cfg = {
        'symbol': 'BTCUSDT',
        'interval': '4h',
        'start_date': '2019-01-01',
        'end_date': '2025-02-25',
        'rebound_pct': 2.0,
        'rsi_min': 30,
        'rsi_max': 40,
        'tp_rsi': 70,
        'time_stop_bars': 30,
        'div_expiry_bars': 60,
        'swing_window': 5,
        'ema52_period': 52,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'rsi_period': 14,
        'fee_pct': 0.06,
    }

    df = fetch_binance_ohlcv(cfg['symbol'], cfg['interval'],
                              cfg['start_date'], cfg['end_date'])
    print("⚙️  計算技術指標...")
    df['macd_line'], df['signal_line'], df['histogram'] = calc_macd(
        df['close'], cfg['macd_fast'], cfg['macd_slow'], cfg['macd_signal'])
    df['rsi']  = calc_rsi(df['close'], cfg['rsi_period'])
    df['ema52'] = calc_ema(df['close'], cfg['ema52_period'])
    print("🔍 偵測 MACD 底背離（正確定義：EMA52 回踩 + MACD 穿越零軸）...")
    df = detect_bullish_divergences(df, cfg['swing_window'],
                                     max_low1_lookback=cfg.get('max_low1_lookback', 200))
    n_div = df['div_signal'].sum()
    print(f"   → 找到 {n_div} 個底背離訊號")
    print("🚀 執行回測...")
    trades = run_backtest(df, cfg)
    report(trades, df)

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    out_csv  = os.path.join(_RESULTS_DIR, 'btc_macd_divergence_trades.csv')
    out_xlsx = os.path.join(_RESULTS_DIR, 'btc_macd_divergence_trades.xlsx')
    out_md   = os.path.join(_RESULTS_DIR, 'btc_macd_divergence_report.md')
    trades.to_csv(out_csv, index=False)
    # Markdown 報告
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
