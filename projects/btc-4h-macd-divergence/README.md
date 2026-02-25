# 子專案：BTC 4H MACD 底背離策略

本子專案對應 **一項獨立研究**：BTC/USDT 4 小時 K 線、MACD 底背離進場策略的驗證與回測。

---

## 策略摘要

| 項目 | 內容 |
|------|------|
| 標的 | BTCUSDT |
| 週期 | 4H |
| 進場 | MACD 4H 底背離確認 → 底部反彈 2% → RSI 30–40 |
| 止盈 | 4H RSI ≥ 70 |
| 止損 | 底背離低點（略下方 0.1% buffer） |
| 時間止損 | 入場後 30 根 4H 棒 |
| 回測區間 | 2019-01-01 → 2025-02-25 |

---

## 回測結果摘要（v1）

- **勝率**：51.2%
- **盈虧比**：1.11x
- **每筆期望值**：約 +0.31%
- **獲利因子**：1.17
- **完成交易**：43 筆
- **出場分布**：止盈 14% / 止損 23% / 時間止損 63%

詳見 `results/` 內 CSV 與主專案 `run.py` 的完整報告輸出。

---

## 本子專案目錄

```
btc-4h-macd-divergence/
├── README.md                      ← 本說明與結論摘要
├── config.json                    ← 本研究所用設定（與主專案 config/ 同步用）
├── btc_divergence_backtest.py     ← 獨立回測腳本（單檔可跑，輸出到 results/）
├── results/                       ← 此研究的回測結果（CSV + XLSX）
│   ├── btc_macd_divergence_trades.csv / .xlsx   ← 獨立腳本輸出
│   └── BTCUSDT_4h_btc_macd_bullish_divergence_v1.csv / .xlsx  ← 主專案 run.py 輸出
└── chat-archive/                  ← 聊天存檔（見下方說明）
```

- **獨立腳本**：在子專案目錄執行 `python btc_divergence_backtest.py` 會抓資料、跑回測、寫入 `results/btc_macd_divergence_trades.csv`。
- **主專案流程**：在 kline-research 根目錄執行 `python run.py --config projects/btc-4h-macd-divergence/config.json` 會寫入 `results/BTCUSDT_4h_...csv`。

---

## 如何繼續此研究

- **改參數再跑**：改 `config.json` 後，在**主專案根目錄**執行：
  ```bash
  python run.py --config projects/btc-4h-macd-divergence/config.json
  ```
  可將 `run.py` 的 `results_dir` 指向本子專案 `results/`，或手動把新 CSV 複製到本子專案 `results/`。
- **沿用舊討論**：在新聊天中說「調用 btc-4h-macd-divergence 的聊天記錄」或 @ 引用 `chat-archive/` 內存檔，再說明要修改的內容。

---

## 聊天存檔

每次使用**新聊天窗口**時，若本研究的討論有延續價值，請把該聊天的摘要或匯出存到 `chat-archive/`，並在下方「記錄索引」補一筆，方便之後用「調用過去的聊天記錄」讓 AI 讀取。

- 存檔方式與引用說明見：**`chat-archive/README.md`**
- 記錄索引：**`chat-archive/INDEX.md`**（可自行維護）
