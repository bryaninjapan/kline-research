# K-Line Trading Research Framework

把 TradingView 看圖直覺轉換成可量化驗證的交易策略研究框架。

---

## 快速開始

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 執行預設策略（BTC MACD 底背離，2019-2025）
python run.py

# 3. 換策略 / 換標的
python run.py --config config/my_strategy.json
```

## 已驗證策略

| 策略 | 標的 | 週期 | 勝率 | 期望值 | 備注 |
|------|------|------|------|--------|------|
| MACD 底背離 | BTC/USDT | 4H | 51.2% | +0.34%/筆 | 止損=底背離低點 |

## 換標的 / 換週期 / 換策略

**標準做法**：複製既有 config，只改必要欄位再跑，不要手寫整份。

| 需求     | 作法 |
|----------|------|
| 換標的   | 複製 `config/btc_macd_divergence.json` → 改 `data.symbol`（如 `ETHUSDT`）、`strategy_name` |
| 換週期   | 同上，改 `data.interval`（如 `1h`、`1d`）；若改為 1d，注意 `time_stop_bars` 等於天數 |
| 新策略   | 從 `config/template.json` 複製，或從現有 config 複製後改 `entry` / `exit` |

然後執行：`python run.py --config config/新檔名.json`。  
AI Agent 詳細流程見 `CLAUDE.md` 的「換標的/換週期/換策略」一節。

## 新增策略

1. `cp config/template.json config/my_strategy.json`
2. 編輯 `my_strategy.json`，填入你的進出場條件
3. `python run.py --config config/my_strategy.json`

完整建模規則請見 `rules/MODELING_RULES.md`。

## 固定流程：子專案與聊天存檔

- **一項研究 = 一個子專案**，放在 `projects/<研究代號>/`（例：`btc-4h-macd-divergence`）。
- 子專案內含：`README.md`（摘要與結論）、`config.json`、`results/`、`chat-archive/`（聊天存檔與索引）。
- 每次你會開**新聊天**；若要**延續或修改**舊研究，說「調用過去的聊天記錄」並 @ 該子專案的 `chat-archive/` 或具體存檔。
- **新研究**時由 AI **建立新子專案**；**舊研究**只改該子專案內容，不新建。

詳見 `CLAUDE.md` 的「固定流程：子專案與聊天存檔」。

## 目錄結構

```
kline-research/
├── CLAUDE.md              ← Claude Code / AI Agent 使用說明
├── README.md
├── run.py                 ← 主入口
├── requirements.txt
├── rules/
│   └── MODELING_RULES.md  ← RULE-0 到 RULE-10 完整規則集
├── config/
│   ├── template.json      ← 新策略模板
│   └── btc_macd_divergence.json
├── src/
│   ├── data_fetcher.py    ← Binance OHLCV 抓取
│   ├── indicators.py      ← MACD, RSI, ATR, Bollinger...
│   ├── signal_detectors.py← 底背離 / 頂背離偵測
│   ├── backtest_engine.py ← 通用回測引擎
│   └── reporter.py        ← 標準化報告輸出
├── skills/
│   └── continuous-learning-v2/  ← 內置學習 skill（可學習換標的/換策略習慣）
├── projects/              ← 子專案：一項研究一個資料夾
│   └── btc-4h-macd-divergence/  ← 例：BTC 4H MACD 底背離
│       ├── README.md
│       ├── config.json
│       ├── results/
│       └── chat-archive/  ← 聊天存檔 + INDEX.md
└── results/               ← 主目錄最新回測輸出（可選）
```

## 核心設計原則

- **無未來資訊洩漏**：所有訊號偵測均有防洩漏保護
- **含交易成本**：手續費 + 滑點是一等公民
- **年份分拆**：報告必含牛市/熊市/震盪各年份表現
- **三種出場**：止盈 + 止損 + 時間止損，缺一不可

詳見 `rules/MODELING_RULES.md`。
