# divergence-detector 集成至 kline-research/projects 的對話存檔

本檔記錄將 divergence-detector 專案規範化集成到 kline-research/projects/ 結構的完整過程。

---

## 一、問題背景

**目標**：把獨立的 divergence-detector 項目添加到 kline-research/projects 的新子項目中，按照 modelling_rules.md 規範組織。

**關鍵約束**：
- 遵循 kline-research 的項目模板規範
- 按照 MODELING_RULES.md（RULE-0 到 RULE-10）逐條驗證
- divergence-detector 保持原始代碼不改動（保留獨立脚本）
- 新增規範化文檔（README、config、MODELING_RULES_COMPLIANCE）

---

## 二、決策過程

### 2.1 三個集成方案評估

#### 方案 A：保留獨立（最小改動）✅ **最終選擇**
- divergence-detector 保持原樣，複製到 projects/ 目錄
- 新增 README.md、config.json、MODELING_RULES_COMPLIANCE.md
- 工作量：< 4 小時
- 優點：零改源碼、學習成本低、無依賴衝突

#### 方案 B：深度集成到 kline-research（大改）
- 修改 kline-research/src/ 的核心庫（signal_detectors.py、backtest_engine.py）
- 工作量：2-3 週、風險高
- 缺點：修改核心文件可能影響其他策略

#### 方案 C：適配層模式（推薦但延後）
- 為 divergence-detector 創建 adapter.py 和 JSON config
- 工作量：1 週
- 缺點：兩個項目仍邏輯分離

### 2.2 最終決策

| 項 | 決策 | 理由 |
|----|------|------|
| **集成方案** | **方案 A：保留獨立** | 最小改動，快速上線，無代碼耦合 |
| **MODELING_RULES** | **需要補充驗證** | 按 RULE-1～RULE-10 逐條驗證，補充缺失文檔 |
| **項目命名** | `btc-eth-4h-cross-divergence` | 符合命名規範 `<symbol>-<interval>-<strategy>` |
| **結果處理** | 刪除，從零開始 | 不保留 output/ 舊數據，集成後重新運行 |

---

## 三、代碼兼容性分析摘要

### 3.1 可複用部分（50%）
✅ Swing point 檢測、頸線計算、交易成本處理、基本報告格式

### 3.2 不可複用部分（50%）
❌ 跨資產背離檢測（需 2 個 DataFrame 同步）、failure_price 定制算法、measured_move 止盈邏輯、rejection 監控條件

**根本原因**：
- divergence-detector：跨資產（BTC+ETH）協調設計
- kline-research：單資產設計
- 參數模式：divergence-detector 用 CLI args，kline-research 用 JSON config

---

## 四、最終集成結構

```
kline-research/projects/btc-eth-4h-cross-divergence/
├── README.md                              # 策略摘要 + 快速開始
├── config.json                            # 參數說明文檔
├── requirements.txt                       # 復製自 divergence-detector
├── .gitignore                             # 復製自 divergence-detector
│
├── scripts/                               # 復製所有 Python 脚本（不改）
│   ├── fetch_data.py
│   ├── detect_divergence.py
│   ├── generate_report.py
│   └── divergenceplus.py
│
├── references/                            # 復製規則文檔 + 新增驗證
│   ├── divergence-rules.md
│   ├── exit-risk-rules.md
│   └── MODELING_RULES_COMPLIANCE.md       # 新增：RULE-1～10 驗證清單
│
├── results/                               # 脚本輸出目錄（初始為空）
│   └── README.md                          # 說明此目錄用途
│
└── chat-archive/                          # 聊天存檔系統
    ├── README.md                          # 存檔說明
    ├── INDEX.md                           # 存檔索引
    └── 2026-02-26-integration-to-kline-research.md  # 本檔
```

---

## 五、關鍵文檔說明

### 5.1 README.md（策略首頁，新人必讀）

**包含內容**：
- 策略摘要表（標的、週期、進出場規則）
- 核心規則總結（背離定義、頸線結構、拒絕確認、資金管理）
- 快速開始指南（首次回測 + 每週更新命令）
- 規則文檔與 MODELING_RULES 合規性的連結

### 5.2 config.json（參數說明文檔，非運行時配置）

**包含內容**：
- 數據參數：symbol、exchange、interval、date range
- 檢測參數：swing_lookback、max_divergence_days、eth_lookback_days 等（每個參數都有註釋）
- 回測參數：sl_buffer_pct、rr_ratio、time_stop_bars、use_measured_move
- 交易成本：fee_pct、slippage_pct
- CLI 示例：便於複制粘貼運行

### 5.3 MODELING_RULES_COMPLIANCE.md（規範驗證清單）

**驗證覆蓋**：RULE-0 ~ RULE-10

**驗證結果摘要**：
| 規則層級 | 狀態 | 說明 |
|---------|------|------|
| RULE-0 ~ RULE-5 | ✅ 符合 | 邏輯清晰，規則完整，無垃圾 |
| RULE-6 驗證 | ⚠️ 部分 | 缺少 Walk-Forward 和年份分拆 |
| RULE-7 回測 | ✅ 符合 | 包含成本，覆蓋多個市場環境 |
| RULE-8 MLOps | ❌ 缺失 | 無實驗追蹤和版本管理 |
| RULE-9 部署 | ⚠️ N/A | 當前為研究項目，非實盤部署 |
| RULE-10 迭代 | ❌ 缺失 | 無正式迭代流程 |

**短期改進優先級**：
1. RULE-6：補充年份分拆數據（divergenceplus.py 已支持）
2. RULE-7：生成完整績效曲線圖
3. RULE-8：初始化實驗日誌範本（在 chat-archive/）

---

## 六、執行清單

### 已完成 ✅
- [x] 代碼兼容性分析（50% 可複用）
- [x] 三個集成方案評估與決策
- [x] 項目目錄結構創建
- [x] 複製所有 scripts、references、requirements.txt、.gitignore
- [x] 規劃文檔內容（README、config、MODELING_RULES_COMPLIANCE）

### 待執行
- [ ] 編寫 README.md
- [ ] 編寫 config.json
- [ ] 編寫 MODELING_RULES_COMPLIANCE.md
- [ ] 初始化 chat-archive/（README.md、INDEX.md）
- [ ] 驗證所有文件和目錄結構完整

---

## 七、如何繼續此研究

### 短期（本週內）
1. 完成上述"待執行"的 4 個文檔
2. 首次運行回測：
   ```bash
   cd projects/btc-eth-4h-cross-divergence
   python scripts/fetch_data.py          # 拉數據
   python scripts/detect_divergence.py   # 檢測背離
   python scripts/generate_report.py     # 生成報告
   python scripts/divergenceplus.py      # 完整回測
   ```
3. 收集回測結果，填入 README.md 和 MODELING_RULES_COMPLIANCE.md

### 中期（1 月內）
4. 補充 RULE-6 的 Walk-Forward 驗證（逐季滾動驗證）
5. 建立實驗日誌系統（記錄每次參數調整的結果）
6. 版本控制：用 git branches 管理參數迭代

### 長期（必要時）
7. 若計劃實盤上線，實現 RULE-9 的熔斷機制和監控
8. 評估是否值得將跨資產背離集成到 kline-research 核心庫（方案 B）

---

## 八、關鍵決策清單

| 決策項 | 選擇 | 原因 |
|--------|------|------|
| 是否修改 divergence-detector 源碼 | 否 | 保留獨立，避免維護負擔 |
| 是否與 kline-research 核心庫集成 | 否，延後 | 跨資產檢測改造工作量大，先驗證策略有效性 |
| 項目存放位置 | kline-research/projects/btc-eth-4h-cross-divergence/ | 符合規範，便於管理多個策略 |
| 結果文件保留 | 刪除舊數據，重新運行 | 與 git 版本控制對齊，避免混亂 |
| MODELING_RULES 驗證範圍 | RULE-1～RULE-10 全覆蓋 | 確保項目質量達標，便於追蹤改進 |

---

## 九、參考資源

- **主規範文檔**：kline-research/rules/MODELING_RULES.md
- **項目模板**：kline-research/projects/btc-4h-macd-divergence/ 與 btc-eth-4h-cross-divergence/
- **原始項目**：/Users/iruka/Downloads/wanai/divergence-detector/
- **規則文檔**：
  - references/divergence-rules.md - 背離檢測規則
  - references/exit-risk-rules.md - 出場與風控規則

---

## 十、常見問題

**Q1：為什麼保留獨立而不完全集成？**
A：完全集成需要修改 kline-research 的核心庫（signal_detectors.py、backtest_engine.py），工作量大且風險高。保留獨立方案讓我們快速上線，同時保持選項開放，若跨資產背離成為常用策略再考慮原生集成。

**Q2：config.json 為什麼不是運行時配置？**
A：divergence-detector 的參數來自 CLI args，無法載入 JSON 運行時。config.json 是參數參考文檔，便於理解和調整。若要 JSON 配置運行時支持，需方案 C 的 adapter.py。

**Q3：MODELING_RULES_COMPLIANCE.md 缺失的部分如何補充？**
A：短期補充 RULE-6（年份分拆表）和 RULE-8（實驗日誌）。中期實現 Walk-Forward 驗證。長期若上線需補充 RULE-9 的熔斷機制。

---

最後更新：2026-02-26
集成方案版本：方案 A（保留獨立）v1.0
MODELING_RULES 版本：v1.0（RULE-0～10 驗證）
