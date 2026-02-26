# BTC-ETH 4H 跨資產背離檢測器

## 策略摘要表

| 項目 | 值 |
|-----|-----|
| **標的** | BTC/ETH USDT 永續合約 (Bybit v5 API) |
| **週期** | 4H K 線 |
| **進場信號** | 跨資產背離 + 頸線突破 + 拒絕確認（3 層確認） |
| **出場規則(止盈)** | Measured Move 或 固定盈虧比 2.0 |
| **出場規則(止損)** | Failure Price ± 0.1% |
| **出場規則(時間)** | 進場後 20 根 4H 強制平倉 |
| **資金管理** | 固定 1% 風險 + 單筆倉位上限 20% |
| **交易成本** | 手續費 0.055% + 滑點 0.05% = 總 0.22% 來回 |
| **回測期間** | [待重新運行] |
| **交易筆數** | [待回測] |
| **勝率** | [待回測] |
| **期望值** | [待回測] |

---

## 核心規則總結

### 背離定義
- **看跌背離**：BTC 創造更高高點（HH），ETH 創造較低高點（LH），5 日內發生
  - 動能衰竭信號 → 趨勢末期
- **看涨背離**：BTC 創造較低低點（LL），ETH 創造較高低點（HL），5 日內發生
  - 反彈信號 → 底部確認

### 三層確認機制
1. **背離形態**：跨資產 swing 點配對
2. **頸線結構**：結構支撐/阻力突破
3. **拒絕確認**：突破後反彈無法觸及失敗價

詳見：`references/divergence-rules.md`

### 出場與風控
- **止損邏輯**：基於 failure_price 設定保守緩衝
- **止盈選項**：Measured Move（結構目標）或固定盈虧比
- **時間止損**：20 根 4H（~80 小時）防止長期持倉

詳見：`references/exit-risk-rules.md`

---

## 快速開始

### 首次完整回測（首次執行 ~50 秒）

```bash
cd projects/btc-eth-4h-cross-divergence

# 安裝依賴（如需）
pip install -r requirements.txt

# Step 1: 從 Bybit 拉取全量 K 線數據（2021-01-01 至今）
python scripts/fetch_data.py

# Step 2: 偵測所有背離模式
python scripts/detect_divergence.py

# Step 3: 生成 Excel 報告（含摘要、看跌、看涨 3 工作表）
python scripts/generate_report.py

# （可選）Step 4: 完整回測（含進出場、資金管理、績效統計）
python scripts/divergenceplus.py

# 結果輸出到 output/ 和 results/ 目錄
```

### 每週增量更新

```bash
# 只拉取上次之後的新 K 線（秒級完成）
python scripts/fetch_data.py --update

# 重新偵測全部背離（毫秒級）
python scripts/detect_divergence.py

# 重新生成 Excel 報告
python scripts/generate_report.py

# 重新執行完整回測
python scripts/divergenceplus.py
```

### 調整參數的例子

```bash
# 調整 Swing 確認窗口和時間範圍
python scripts/detect_divergence.py --swing-lookback 4 --max-days 7

# 調整止盈止損參數
python scripts/divergenceplus.py --rr-ratio 2.5 --time-stop-bars 24 --use-measured-move
```

---

## 完整規則文檔

### 背離檢測規則
- **檔案**：`references/divergence-rules.md`
- **內容**：3 Condition 背離定義、Swing 點檢測算法、頸線結構、拒絕確認邏輯

### 出場與風控規則
- **檔案**：`references/exit-risk-rules.md`
- **內容**：進場時機、止損策略、止盈計算、資金管理、交易成本

### MODELING_RULES 合規性
- **檔案**：`references/MODELING_RULES_COMPLIANCE.md`
- **內容**：按 RULE-0 ~ RULE-10 逐條驗證，記錄缺失部分和改進優先級

---

## 聊天存檔

此項目的所有開發記錄、決策邏輯和實驗日誌保存於 `chat-archive/`。

- `chat-archive/README.md`：存檔系統說明
- `chat-archive/INDEX.md`：所有檔案索引
- `chat-archive/2026-02-26-divergence-integration.md`：本次集成記錄

詳見：`chat-archive/INDEX.md`

---

## 項目結構

```
btc-eth-4h-cross-divergence/
├── README.md                           ← 本檔案
├── config.json                         ← 參數說明
├── requirements.txt                    ← Python 依賴
├── .gitignore                          ← Git 忽略規則
│
├── scripts/                            ← 核心執行腳本
│   ├── fetch_data.py                   # 從 Bybit API 拉取 K 線
│   ├── detect_divergence.py            # 背離檢測引擎
│   ├── generate_report.py              # Excel 報告生成
│   └── divergenceplus.py               # 完整回測 + 風控
│
├── references/                         ← 規則文檔
│   ├── divergence-rules.md             # 背離檢測規則
│   ├── exit-risk-rules.md              # 出場與風控規則
│   └── MODELING_RULES_COMPLIANCE.md    # 規範合規驗證
│
├── output/                             ← 腳本臨時輸出（K 線 CSV、結果 JSON 等）
│
├── results/                            ← 最終回測結果
│   └── README.md                       # 結果目錄說明
│
└── chat-archive/                       ← 開發記錄存檔
    ├── README.md                       # 存檔系統說明
    ├── INDEX.md                        # 檔案索引
    └── 2026-02-26-divergence-integration.md  # 集成記錄
```

---

## 快速參考

| 任務 | 命令 |
|------|------|
| 拉數據 | `python scripts/fetch_data.py` |
| 檢測背離 | `python scripts/detect_divergence.py` |
| 生成 Excel | `python scripts/generate_report.py` |
| 完整回測 | `python scripts/divergenceplus.py` |
| 增量更新 | `python scripts/fetch_data.py --update` |

---

## 後續行動

集成完成後，建議按以下順序進行：

1. **運行首次回測**：執行上述「首次完整回測」命令，驗證環境正常
2. **收集年份分拆數據**：按 RULE-6 要求統計每年的勝率、期望值、市況
3. **實現 Walk-Forward 驗證**：按季滾動驗證策略穩健性
4. **建立實驗日誌**：記錄每次修改參數的動機和結果

詳見：`references/MODELING_RULES_COMPLIANCE.md` 的改進優先級

---

**最後更新**：2026-02-26
**項目狀態**：✅ 集成完成，待首次回測驗證
