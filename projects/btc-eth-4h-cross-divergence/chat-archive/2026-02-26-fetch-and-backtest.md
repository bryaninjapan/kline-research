# 首次數據拉取與完整回測記錄

**日期**：2026-02-26
**任務**：執行 `fetch_data.py` 拉取數據，跑完整回測（detect_divergence → divergenceplus）

---

## 執行摘要

| 步驟 | 指令 | 結果 |
|------|------|------|
| 1. 數據拉取 | `python3 scripts/fetch_data.py` | BTC 11,294 根 4H、ETH 10,856 根 4H（2021 → 2026-02-26） |
| 2. 背離偵測 | `python3 scripts/detect_divergence.py` | Bearish 163（109 成功）、Bullish 150（90 成功） |
| 3. 完整回測 | `python3 scripts/divergenceplus.py --use-measured-move --rr-ratio 2 --time-stop-bars 20` | 313 筆交易，勝率 67.7%，期望值 +2.03% |

---

## 回測結果（整體）

| 項目 | 數值 |
|------|------|
| 交易次數 | 313 |
| 勝率 | 67.7% |
| 平均獲利 / 平均虧損 | +4.525% / -3.210% |
| 盈虧比 | 1.41 |
| 期望值 | +2.0290% |
| Sharpe | 7.499 |
| 最大回撤 | -23.72% |
| 出場分佈 | SL=83 \| TP=127 \| TimeStop=103 |

**Bearish**：163 筆，勝率 68.1%，期望值 +2.20%，出場 SL=42 / TP=71 / TS=50  
**Bullish**：150 筆，勝率 67.3%，期望值 +1.84%，出場 SL=41 / TP=56 / TS=53  

**年份分拆**：2021–2024 期望值為正；2025–2026 較弱（2026 僅 11 筆）。

---

## 程式修正（本次對話中完成）

| 檔案 | 問題 | 修正 |
|------|------|------|
| `scripts/divergenceplus.py` | Python 3.9 不支援 `dict \| None` | 改為 `Optional[dict]`，並 `from typing import Optional` |
| `scripts/divergenceplus.py` | `load_ohlcv` 用 `index_col=0, parse_dates=True` 導致 K 線日期變成 1970-01-01 | 改為讀取 CSV 後用 `timestamp`（ms）轉 `pd.to_datetime(..., unit="ms", utc=True)` 設為 index |
| `scripts/divergenceplus.py` | `future_mask.values` 在 ndarray 上無 `.values` | 改為 `np.argmax(np.asarray(future_mask))` |

---

## 產出檔案位置

- `output/BTCUSDT_4h.csv`、`output/ETHUSDT_4h.csv` — 原始 K 線
- `output/divergence_results.json` — 背離偵測結果
- `output/divergenceplus_trades.csv` — 交易明細
- `output/divergenceplus_report.txt` — 績效報告
- `output/divergenceplus_equity.csv` — 資金曲線

---

## 下一步（可選）

1. 將報告摘要或關鍵 CSV 複製到 `results/` 以符合子專案規範
2. 更新 `MODELING_RULES_COMPLIANCE.md` 填入本次回測結果
3. 後續可做參數敏感度（swing_lookback、time_stop_bars、rr_ratio）或 Walk-Forward 驗證
