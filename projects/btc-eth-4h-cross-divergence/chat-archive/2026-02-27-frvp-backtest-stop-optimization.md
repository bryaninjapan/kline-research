# FRVP 回測腳本建立 + Stop Loss 優化實驗

**日期**：2026-02-27
**任務**：建立全新 `frvp_backtest.py`（FRVP 規則），實驗不同止損模式，並與舊 `divergenceplus.py` 結果比對

---

## 背景

本對話是從前一個 context 壓縮後繼續的。前一個 context 已完成：
- `frvp_backtest.py` 腳本建立（完整 FRVP 規則）
- 數據源從 Bybit 切換至 Binance（ETH Bybit 永續僅從 2021-03-15 才有）
- 初步參數調整（sync_tol=5, min_gap=3, max_gap=16）→ 24 signals, 15 completed trades

本次 context 主要工作：**實驗 stop_mode=prev_high，並解答舊腳本 67.7% 高勝率的原因**

---

## 本次實驗：Stop Loss 模式比較

### 新增 `--stop-mode` 參數

修改 `scripts/frvp_backtest.py`，新增三種止損模式：

| stop_mode | 止損位計算 | 說明 |
|-----------|-----------|------|
| `vah` | `VAH × (1 + buffer%)` | 原始做法（85th pct 收盤） |
| `prev_high` | `high[entry_bar-1] × (1 + buffer%)` | 進場前一根棒的 high（預設改為此） |
| `sh2` | `SH2_price × (1 + buffer%)` | 背離第二高點 |

`DEFAULT_CFG["stop_mode"]` 改為 `"prev_high"`。

### 修改內容（`frvp_backtest.py`）

1. `DEFAULT_CFG` 新增 `"stop_mode": "prev_high"`
2. `_open_trade()` 簽名擴展：加入 `btc_high`, `eth_high`, `stop_mode` 參數
3. 兩個 `_open_trade()` call site 都更新為傳入 `btc_high`, `eth_high`, `cfg["stop_mode"]`
4. `parse_args()` 新增 `--stop-mode {vah,prev_high,sh2}`
5. `main()` cfg dict 加入 `"stop_mode": args.stop_mode`
6. report 風險提示區加入止損模式說明

---

## 回測結果比較

執行指令：
```bash
python3 scripts/frvp_backtest.py --sync-tol 5 --min-gap 3 --max-gap 16 --stop-mode prev_high
```

### VAH 模式（前次結果）vs prev_high 模式（本次）

| 指標 | vah（前次） | prev_high（本次） |
|------|------------|-----------------|
| 完成交易 | 15 | 15 |
| 勝率 | 20.0% | 6.7% |
| 平均盈利 | — | +6.30% |
| 平均虧損 | — | -3.57% |
| 期望值 | -3.317% | -2.914% |
| 止損出場 | 12/15 | 14/15 |
| 平均持倉 | — | 11.8 根（47h） |

### prev_high 模式年份分拆

| 年份 | 筆數 | 勝率 | 期望值 | 市況 |
|------|------|------|--------|------|
| 2019 | 4 | 25.0% | +0.561% | — |
| 2020 | 2 | 0.0% | -3.297% | — |
| 2021 | 2 | 0.0% | -3.335% | 牛市 |
| 2022 | 2 | 0.0% | -6.053% | 熊市 |
| 2023 | 1 | 0.0% | -5.401% | 復甦 |
| 2024 | 2 | 0.0% | -3.795% | 牛市 |
| 2025 | 2 | 0.0% | -3.794% | 牛市? |

### 背離類型分拆（prev_high）

| 類型 | 筆數 | 勝率 | 期望值 |
|------|------|------|--------|
| BTC_HH+ETH_LH | 7 | 14.3% | -1.447% |
| BTC_LH+ETH_HH | 8 | 0.0% | -4.197% |

### 結論

- `prev_high` 止損**更緊**，導致 May 2021 大跌行情（原 +25.83% 大贏單）被回彈掃出，變成 -5.06% 虧損
- `vah` 止損更寬，能撐過回彈並捕捉到大行情
- `BTC_LH+ETH_HH` 類型在兩種模式下均 0% 勝率，根本不適合做空

---

## 解答：為何舊腳本 `divergenceplus.py` 勝率 67.7%？

用戶在對話中提問：「那為什麼之前的回測數據有 60% 以上？」

### 三個核心差異

| 維度 | divergenceplus.py（舊） | frvp_backtest.py（新） |
|------|------------------------|----------------------|
| 方向 | **做多 + 做空** | 只做空 |
| 止盈 | **Measured Move**（頸線結構目標） | BTC RSI < 30 連續 2 根 |
| 時間止損 | ✅ **20 根強制出場** | ❌ 無 |
| 交易次數 | 313 筆 / 5 年 | 15 筆 / 7 年 |
| 年份 | 2021–2026（含大牛市做多） | 2019–2025 |

### 主要解釋

1. **做多方向是主要貢獻**：2021–2024 大牛市，做多訊號勝率天然高，拉高整體勝率
2. **Measured Move TP 容易達到**：只需價格走 20–30% 的路程到結構目標；RSI < 30 要求市場進入極度超賣，在牛市幾乎不會出現
3. **20 根時間止損**：避免無限期持有沒有方向的交易。新腳本無時間止損，交易 #6 持倉 112 根棒（18 天）才止損
4. **舊腳本勝率是合法的**：entry 條件是等 15 根確認 rejection（無未來數據），沒有 look-ahead bias

---

## 當前策略瓶頸

`frvp_backtest.py` 純做空，面臨以下結構性問題：

1. `BTC_LH+ETH_HH`（13 個訊號，8 筆交易）= **0% 勝率**，應全面過濾
2. 在牛市逆勢做空，RSI < 30 止盈幾乎無法達到
3. 只有 2019 年（熊市/震盪）有正期望值

---

## 下一步選項（待用戶選擇）

| 選項 | 說明 |
|------|------|
| A — 過濾 BTC_LH+ETH_HH | 只交易 BTC_HH+ETH_LH（ETH 是弱方），符合策略本意 |
| B — RSI 趨勢過濾 | 只在 BTC RSI > 65 時進場，減少逆勢空單 |
| C — 宏觀過濾 | BTC 跌破 200 根 EMA 才允許做空 |
| D — 借用舊規則 TP | measured move + 20 根時間止損 取代 RSI < 30 |

---

## 當前配置（本對話結束時）

```
stop_mode  = prev_high   ← 本次新增，現為預設
sync_tol   = 5
min_gap    = 3
max_gap    = 16
swing_window = 5
vah_buffer = 0.5%
tp_rsi     = < 30 (2 consecutive bars)
fee        = 0.06%/side
data       = Binance 4H  (2019-01-01 → 2025-12-31)
```

---

## 輸出文件

| 文件 | 路徑 |
|------|------|
| 回測腳本 | `scripts/frvp_backtest.py` |
| 交易記錄 | `results/frvp_trades.csv` |
| 回測報告 | `results/frvp_report.txt` |
| 舊系統報告（對比用） | `output/divergenceplus_report.txt` |
