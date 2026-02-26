# divergence-detector 集成記錄

**日期**：2026-02-26
**任務**：將 `/wanai/divergence-detector` 整合到 `kline-research/projects/btc-eth-4h-cross-divergence/`

---

## 決策摘要

| 項 | 決策 | 理由 |
|----|------|------|
| **集成方案** | 方案 A：保留獨立 | 最小改動，快速上線；divergence-detector 零改動 |
| **MODELING_RULES** | 需要補充驗證 | 按 RULE-1～RULE-10 逐條驗證，補充缺失文檔 |
| **結果處理** | 從零開始 | 不保留 output/ 舊數據，集成後重新運行腳本 |

---

## 三個方案分析

| 方案 | 工作量 | 優點 | 缺點 |
|-----|--------|------|------|
| A：保留獨立 | < 4h | 零改動，快速上線 | 無法復用 kline-research 通用庫 |
| B：深度集成 | 2-3 週 | 完全復用框架 | 風險高，修改核心文件 |
| C：適配層 | 1 週 | JSON 參數化，低耦合 | 功能分散，兩個項目並存 |

**選擇 A 的原因**：divergence-detector 已有完整功能，強行深度集成會引入不必要複雜度。

---

## 兼容性評估

### 可直接復用（~50%）
- Swing point 檢測（find_swing_highs / find_swing_lows）
- 頸線計算（neckline = min/max between two swings）
- 交易成本處理（fee_pct + slippage_pct）
- 基本報告格式

### 不兼容的根本差異（~50%）
1. **跨資產協調**：2-symbol 同步邏輯 vs kline-research 單資產設計
2. **參數模塊化**：80% 參數硬編碼或 CLI args（無 JSON config）
3. **進出場定制**：divergenceplus.py 完全圍繞背離設計，難以通用化

---

## 執行紀錄

### 已存在（集成前已部分完成）
- [x] scripts/ — 4 個 .py 檔案（與 divergence-detector 同步）
- [x] references/ — divergence-rules.md, exit-risk-rules.md
- [x] requirements.txt, README.md, .gitignore
- [x] output/, results/, chat-archive/ 目錄

### 本次新增
- [x] config.json — 參數說明文檔（v1.0）
- [x] references/MODELING_RULES_COMPLIANCE.md — RULE-0 ~ RULE-10 驗證
- [x] chat-archive/README.md — 存檔系統說明
- [x] chat-archive/INDEX.md — 存檔索引
- [x] chat-archive/2026-02-26-divergence-integration.md — 本記錄
- [x] results/README.md — 結果目錄說明

---

## MODELING_RULES 主要缺口

| 規則 | 狀態 | 待辦 |
|------|------|------|
| RULE-6 Walk-Forward | ⚠️ 未實現 | 實現逐季滾動驗證 |
| RULE-6 年份分拆 | ⚠️ 待補充 | 按年統計勝率/期望值 |
| RULE-8 實驗追蹤 | ❌ 缺失 | 建立實驗日誌 |
| RULE-10 迭代流程 | ❌ 缺失 | 建立版本控制和參數日誌 |

---

## 下一步行動

1. 執行首次完整回測，驗證環境正常
   ```bash
   cd projects/btc-eth-4h-cross-divergence
   python scripts/fetch_data.py
   python scripts/detect_divergence.py
   python scripts/generate_report.py
   python scripts/divergenceplus.py
   ```

2. 收集年份分拆數據（RULE-6）
3. 實現 Walk-Forward 驗證
4. 更新 MODELING_RULES_COMPLIANCE.md 填入實際回測結果
