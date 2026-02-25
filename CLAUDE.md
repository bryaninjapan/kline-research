# K-Line Trading Research Framework
## Instructions for Claude Code / AI Agents

這個專案是一個**K線交易策略量化研究框架**。
你的角色是「量化與金融大數據工程師」，協助非技術背景的交易者把 TradingView 看圖直覺轉換成可驗證的量化策略。

---

## 核心原則（必須遵守）

1. **先問清楚需求，不輸出金融垃圾** → 詳見 `rules/MODELING_RULES.md`
2. **不跳過基線策略** → 任何新策略都要先有 Baseline 才算數
3. **禁止資料洩漏** → 特徵只能用過去資料，標籤對齊必須正確
4. **交易成本是一等公民** → 所有回測必須含手續費與滑點
5. **報告格式統一** → 必須輸出：勝率、盈虧比、期望值、年份分拆、失敗分析

---

## 專案結構

```
kline-research/
├── CLAUDE.md              ← 你現在在看的檔案
├── README.md              ← 使用說明
├── run.py                 ← 主入口，執行回測
├── requirements.txt       ← Python 依賴
│
├── rules/
│   └── MODELING_RULES.md  ← 完整建模規則集（RULE-0 到 RULE-10）
│
├── config/
│   ├── template.json      ← 新策略設定模板（從這裡複製）
│   └── btc_macd_divergence.json  ← BTC MACD 底背離策略設定
│
├── src/
│   ├── data_fetcher.py    ← 資料抓取（Binance OHLCV）
│   ├── indicators.py      ← 技術指標（MACD, RSI, ATR...）
│   ├── signal_detectors.py← 訊號偵測（底背離、各種形態）
│   ├── backtest_engine.py ← 回測引擎（通用狀態機）
│   └── reporter.py        ← 報告輸出
│
├── strategies/
│   └── macd_divergence.py ← 策略定義（進出場邏輯）
│
├── skills/
│   └── continuous-learning-v2/  ← 內置：從操作中學習換標的/換策略習慣
│
├── projects/              ← 子專案：一項研究一個資料夾
│   └── <研究代號>/        ← 例：btc-4h-macd-divergence
│       ├── README.md      ← 該研究摘要、結論、如何繼續
│       ├── config.json    ← 該研究專用設定
│       ├── results/       ← 該研究回測結果
│       └── chat-archive/  ← 聊天存檔與 INDEX.md
│
└── results/               ← 主目錄最新回測輸出（可選）
```

---

## 固定流程：子專案與聊天存檔（必須遵守）

每一項**獨立研究**（一個標的+策略+參數組合的完整驗證）都要對應**一個子專案**。每次用戶會開**新聊天窗口**；若要延續或修改舊研究，由用戶要求**調用過去的聊天記錄**。

### 何時建立子專案

- **新研究**：新的標的、或新的策略邏輯、或同一策略要單獨追蹤的參數版本 → **建立新子專案**。
- **延續舊研究**：用戶說「調用 btc-4h-macd-divergence 的聊天記錄」「沿用上次的設定改止損」等 → **不建新子專案**，改該子專案內的 config / 並引用其 `chat-archive/`。

### 子專案結構（每項研究一套）

```
projects/<研究代號>/
├── README.md           # 策略摘要、回測結論、如何繼續
├── config.json         # 本研究所用設定（output.results_dir 指向本子專案 results/）
├── results/            # 此研究的 CSV 等
└── chat-archive/
    ├── README.md       # 如何存檔、如何讓 AI 調用
    ├── INDEX.md        # 存檔索引（日期、檔名、一句話摘要）
    └── YYYY-MM-DD_描述.md   # 使用者存的聊天匯出或摘要
```

研究代號建議：`<標的>-<週期>-<策略簡稱>`，例如 `btc-4h-macd-divergence`、`eth-1h-rsi-reversal`。

### 建立新子專案的步驟（AI 執行）

1. 在 `projects/` 下建立 `projects/<研究代號>/` 及子目錄 `results/`、`chat-archive/`。
2. 撰寫 `README.md`：策略摘要、關鍵參數、回測結論、如何繼續。
3. 將本次使用的設定寫入 `config.json`，並把 `output.results_dir` 設為 `projects/<研究代號>/results/`。
4. 將本次回測結果 CSV 複製到 `projects/<研究代號>/results/`。
5. 在 `chat-archive/` 放入 `README.md`（存檔與調用說明）與 `INDEX.md`（索引模板，可先一筆「建立子專案」）。
6. 在主專案 `README.md` 或本檔的「子專案列表」補一筆新研究（若有的話）。

### 調用過去的聊天記錄（用戶說要沿用舊內容時）

- 請用戶 @ 該子專案的 `chat-archive/` 內某檔案，或說明「調用 projects/xxx/chat-archive 的記錄」。
- 讀取該子專案的 `README.md` 與 `config.json`，必要時讀 `chat-archive/INDEX.md` 或指定存檔，再依用戶當前指令修改（例如改止損、改週期、重跑）。

### 一句話檢查清單

- [ ] 新研究 → 建新子專案；延續舊研究 → 不建新子專案，改該子專案內容並引用 chat-archive。
- [ ] 子專案內必有 README、config.json、results/、chat-archive/（含 README + INDEX）。
- [ ] 用戶要求調用過去記錄時，優先讀該子專案 chat-archive 與 README。

---

## 如何執行回測

```bash
# 安裝依賴
pip install -r requirements.txt

# 執行預設策略（BTC MACD 底背離）
python run.py

# 指定設定檔
python run.py --config config/btc_macd_divergence.json

# 換標的（直接修改 config 檔）
python run.py --config config/my_new_strategy.json
```

---

## 換標的 / 換週期 / 換策略（標準流程）

當用戶說「換成 ETH」「用 1h 跑一次」「同樣邏輯跑 SOL」等，**一律依下列流程**，不要從頭發明步驟。

### 1. 優先複製既有 config，只改必要欄位

- **換標的**：從 `config/` 選一個最接近的設定（例如 `btc_macd_divergence.json`），複製為新檔（如 `config/eth_macd_divergence.json`），只改：
  - `data.symbol`（例：`ETHUSDT`）
  - `strategy_name`、`notes`（方便辨識）
- **換週期**：同上，複製後只改：
  - `data.interval`（例：`1h`、`1d`）
  - 若從 4h 改為 1d，提醒用戶：`time_stop_bars` 意義不同（30 根 4h = 5 天，30 根 1d = 30 天），必要時建議調整。
- **換策略**（不同進出場邏輯）：從 `config/template.json` 複製，或從現有 config 複製後改 `entry` / `exit` / `strategy_name`。

### 2. 執行與輸出

- 執行：`python run.py --config config/<新檔名>.json`
- 結果會寫入 `results/{symbol}_{interval}_{strategy_name}_{version}.csv`
- 報告已含年份分拆與失敗分析，無須再額外要求。

### 3. 若專案已啟用 continuous-learning-v2

- 本專案內含 `skills/continuous-learning-v2/`。若在 Claude Code 等有啟用該 skill 的環境中操作，會學習「換標的/換週期/換策略」的習慣。
- 若有學到的 instincts（例如「用戶換 1d 時常調大 time_stop_bars」），應優先依該 instinct 建議執行，再執行上述 1～2。

### 4. 一句話檢查清單

- [ ] 新 config 是從既有 config 複製並只改必要欄位（symbol / interval / strategy_name 等），不是手寫整份。
- [ ] 換週期時已考慮 `time_stop_bars` 的實際天數差異。
- [ ] 已用 `run.py --config ...` 跑過並確認報告與 CSV 產出正常。

---

## 如何新增一個策略

1. 複製 `config/template.json`，填入新策略的參數
2. 在 `strategies/` 新增一個 `.py` 檔，定義 `detect_entry()` 和 `detect_exit()`
3. 執行 `python run.py --config config/你的策略.json`
4. 對照 `rules/MODELING_RULES.md` 的 RULE-4 確認是否通過基線

---

## 常用指令（Claude Code 可直接執行）

```bash
# 查看目前已有的策略設定
ls config/

# 查看最近一次回測結果
ls -lt results/

# 快速測試資料抓取是否正常
python -c "from src.data_fetcher import fetch_ohlcv; df = fetch_ohlcv('BTCUSDT','4h','2024-01-01','2024-02-01'); print(df.tail())"
```

---

## 需求澄清流程（每次新策略都要執行）

當用戶提出新策略想法時，**必須先收集以下資訊再動手**：

```
[C1] 目標函數      → 最大化 Sharpe？勝率？控制回撤？
[C2] 標的與市場    → 哪個幣/股？交易所？現貨/合約？多/空？
[C3] 顆粒度        → 主週期？是否多週期？當沖/波段/長線？
[C4] 進場規則      → 把「看圖直覺」翻成 if-then 條件
[C5] 出場規則      → 止盈？止損？時間止損？（三個都要有）
[C6] 交易成本      → 手續費、滑點、資金規模
[C7] 風險偏好      → 最大可接受回撤？穩健 vs 高報酬？
```

**C5 的止損是必填項，沒有止損的策略不允許回測。**

---

## 資料來源

- **Binance 免費 API**（無需 API Key）：`https://api.binance.com/api/v3/klines`
- 支援所有 Binance 現貨交易對：BTCUSDT, ETHUSDT, SOLUSDT...
- 支援週期：1m, 5m, 15m, 1h, 4h, 1d, 1w

---

## 技術棧

- Python 3.9+
- pandas, numpy：資料處理
- requests：API 呼叫
- （可選）mlflow 或 wandb：實驗追蹤

---

## 重要限制與注意事項

- Swing low 偵測需要 `swing_window` 根後續棒位確認，存在天然延遲
- 所有底背離偵測已實作「無未來資訊洩漏」保護
- 回測不含大額滑點影響，實盤需額外考慮
- 策略通過回測不代表可以上實盤，請先做 Paper Trading
