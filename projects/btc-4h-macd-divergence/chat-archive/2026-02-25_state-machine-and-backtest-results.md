# BTC 4H MACD 底/頂背離 — 狀態機規則與回測結果

日期：2026-02-25
涵蓋：兩個獨立腳本的狀態機設計、關鍵參數、做空機制、回測結果摘要

---

## 腳本檔案

- **Bullish (多):** `btc_divergence_backtest.py`
- **Bearish (空):** `btc_bearish_divergence_backtest.py`

兩者位於 `kline-research/projects/btc-4h-macd-divergence/`

---

## Bullish 狀態機（detect_bullish_divergences）

每遇到新的 swing low，與 Low1 做比較，分三個分支：

**1. No recovery**（Low1 和新 SL 之間 MACD 從未穿越零軸）
- 新 SL 價格 ≤ Low1 價格 → 更新 Low1（保留更低的錨點）
- 否則 → 保持 Low1 不動

**2. Recovery + divergence found**
- 標記訊號，Low1 = sl_pos（Low2 成為新的 Low1）

**3. Recovery + no divergence**
- 新 SL 價格 < Low1 價格 → 更新 Low1
- Elif Low1 過舊（sl_pos - low1_pos > max_low1_lookback=200 棒）→ 更新 Low1
- Else → **保持舊 Low1**（防止淺 "price HL + MACD HL" 覆蓋更深的錨點）

**為什麼 max_low1_lookback=200？**
- 2024-08-05 極端崩跌（$51k, MACD=-2791）讓 Low1 被鎖死好幾個月
- 200棒 ≈ 33天：Aug5→Oct2 = ~348棒（過舊）→ 重設 ✓
- Nov26→Dec20 = 142棒（未過舊）→ Nov26 保留為 Low1 → Dec20 隱藏背離被偵測 ✓

**Recovery 條件（bullish）：**
- MACD line 和 signal line 都穿越到零軸以上
- 價格觸及 EMA52（close[j] >= ema52[j]）

---

## Bearish 狀態機（detect_bearish_divergences）

Bullish 的鏡像 — 追蹤 High1 錨點，只考慮 MACD > 0 的 swing high。

**1. No recovery**（High1 和新 SH 之間 MACD 從未穿越零軸以下）
- 新 SH 價格 ≥ High1 價格 → 更新 High1（保留更高的錨點）
- 否則 → 保持 High1 不動

**2. Recovery + divergence found**
- 標記訊號，High1 = sh_pos

**3. Recovery + no divergence**
- 新 SH 價格 > High1 價格 → 更新 High1
- Elif High1 過舊（> max_high1_lookback=200 棒）→ 更新 High1
- Else → **保持舊 High1**

**Recovery 條件（bearish）：**
- MACD line 和 signal line 都穿越到零軸以下
- 價格向下觸及 EMA52（close[j] <= ema52[j]）

---

## 兩種背離類型

**Bullish (多)：**
- Regular：Price LL + MACD HL（MACD 負值較小）→ 趨勢反轉訊號
- Hidden：Price HL + MACD LL（MACD 負值更深）→ 多頭延續訊號

**Bearish (空)：**
- Regular：Price HH + MACD LH（MACD 正值較小）→ 趨勢反轉訊號
- Hidden：Price LH + MACD HH（MACD 正值更大）→ 空頭延續訊號

---

## 關鍵參數

**Bullish cfg：**
```python
'div_expiry_bars': 60,     # 訊號有效期 60 棒（~10 天）
'max_low1_lookback': 200,  # Low1 超過 200 棒視為過舊（~33 天）
'swing_window': 5,
'ema52_period': 52,
'rsi_min': 30, 'rsi_max': 40,  # 入場 RSI 範圍（從超賣回升）
'rebound_pct': 2.0,            # 距 div_low 回彈 2% 才入場
'tp_rsi': 70,
# 無 time_stop（已移除）
```

**Bearish cfg：**
```python
'div_expiry_bars': 60,
'max_high1_lookback': 200,
'swing_window': 5,
'ema52_period': 52,
'rsi_min': 60, 'rsi_max': 70,  # 做空入場 RSI 範圍（從超買下降）
'rebound_pct': 2.0,            # 距 div_high 下跌 2% 才入場
'tp_rsi': 30,                  # 止盈 RSI
'fee_pct': 0.06,
```

---

## RSI 入場條件

同時檢查**當前棒**或**前一棒**：
```python
rsi_ok = (rsi_min <= rsi[i] <= rsi_max) or (rsi_min <= rsi[i-1] <= rsi_max)
```
允許：RSI 在 N 棒符合範圍，N+1 棒達到 2% 條件再入場。

---

## div_low / div_high 計算

- Bullish：`div_low = min(low_arr[sl_pos : confirmed_pos+1])` — 5 棒確認窗口內的最低 bar-low
- Bearish：`div_high = max(high_arr[sh_pos : confirmed_pos+1])` — 5 棒確認窗口內的最高 bar-high

---

## 做空交易機制（Bearish）

| 項目 | 說明 |
|------|------|
| 入場條件 | `close[i] <= div_high * (1 - 0.02)` 且 RSI 60–70 |
| 入場價 | `entry_price = close[i] * (1 - fee)`（做空賣出） |
| 止損設定 | `stop_loss = div_high * 1.001`（稍高於 div_high） |
| 止損觸發 | `high[i] >= stop_loss` |
| 止損出場 | `exit_px = max(stop_loss, close[i]) * (1 + fee)` |
| 止盈出場 | `exit_px = close[i] * (1 + fee)`（RSI ≤ 30 時） |
| PnL 計算 | `(entry_price - exit_px) / entry_price * 100` |

---

## 回測結果（2019-01-01 to 2025-02-25）

**Bullish（無時間止損）：**
- 26 筆完成交易
- 勝率 19.2%，平均獲利 +7.74%，平均虧損 -5.53%，R/R 1.40x
- 期望值 -2.980%，Profit Factor 0.33

**Bearish：**
- 24 筆完成交易
- 勝率 37.5%，平均獲利 +5.99%，平均虧損 -4.93%，R/R 1.21x
- 期望值 -0.838%，Profit Factor 0.73
- 年份分拆：
  - 2022（熊市）：50% 勝率，期望值 +2.33%
  - 2023（復甦）：80% 勝率，期望值 +2.54%
  - 2024（牛市）：0% 勝率，平均虧損 -4.93%

---

## 輸出檔案路徑

```
kline-research/projects/btc-4h-macd-divergence/results/
├── btc_macd_divergence_trades.csv       # Bullish 交易明細
├── btc_macd_divergence_report.md        # Bullish 報告
├── btc_macd_bearish_trades.csv          # Bearish 交易明細
└── btc_macd_bearish_report.md           # Bearish 報告
```
