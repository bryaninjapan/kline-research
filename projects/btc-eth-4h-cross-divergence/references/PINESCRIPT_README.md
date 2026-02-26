# TradingView Pine Script v6 使用說明

專案內提供兩個 Pine Script v6 腳本，對應 `divergence-rules.md` 與 `exit-risk-rules.md` 的進出場邏輯，可在 TradingView 上查看訊號與回測。

## 檔案位置

| 檔案 | 類型 | 用途 |
|------|------|------|
| `scripts/CrossAssetDivergence.pine` | **indicator** | 僅看圖、標記進場點、頸線/止損參考、警報 |
| `scripts/CrossAssetDivergence_Strategy.pine` | **strategy** | 回測：含進場、止損、止盈、時間止損 |

## 使用步驟（Binance 範例）

1. **圖表設定（必做）**  
   - 標的：**BINANCE:ETHUSDT** 或 **BINANCE:BTCUSDT**（二選一即可，腳本會自動拉另一標的）  
   - 週期：**4H**（上方週期選 4H，策略設計以 4H 為準）  
   - 腳本會用**圖表同一週期**請求另一標的，所以圖表 4H = 請求 4H，K 線才會對齊、有數據。

2. **加入腳本**  
   - 開啟 TradingView →  Pine 編輯器 → 新增腳本  
   - 複製 `CrossAssetDivergence.pine` 或 `CrossAssetDivergence_Strategy.pine` 全文貼上  
   - 儲存後加到圖表（Indicator 或 Strategy 擇一或都加）。

3. **策略屬性（Strategy Tester 設定）**  
   - **Initial capital**：例如 10000（不影響「沒有數據」問題）  
   - **Default order size**：10% of equity 或固定張數皆可  
   - **Commission / Slippage**：可設 0.055% 與若干 ticks 貼近實盤  

4. **腳本參數（與 config.json 對齊）**  
   - **依圖表自動選另一標的**：建議**開啟**（BINANCE:ETH → 自動請求 BINANCE:BTC，反之亦然）  
   - **Swing 確認棒數**：3  
   - **H1–H2 最大間隔**：30（約 5 日）  
   - **頸線突破後監控根數**：15  
   - **止損緩衝 %**：0.1  
   - **時間止損 (4H 根數)**：20  
   - **止盈**：Measured Move 或固定盈虧比 2  
   - **另一標的**：關閉自動時手動填 BINANCE:BTCUSDT 或 BINANCE:ETHUSDT

## 邏輯對應

| 規則 | 腳本實作 |
|------|----------|
| 進場 | Rejection 確認後**下一根 4H 收盤**進場（多/空） |
| 止損 | failure_price × (1 ± 0.1%)，收盤穿越即出場 |
| 止盈 | Measured Move（頸線到 H2/L2 距離投影）或固定 RR |
| 時間止損 | 進場後 20 根 4H 未達 TP/SL 則市價平倉 |

## 「Not enough data to display」排除

若加載到 **ETHUSDT** 或 **BTCUSDT** 後出現 "Not enough data to display"，常見原因與處理：

| 原因 | 處理方式 |
|------|----------|
| **圖表與請求的交易所不一致** | 開啟參數「依圖表自動選另一標的」：腳本會用圖表同一交易所請求另一標的（圖表 BINANCE:ETH → 請求 BINANCE:BTC），K 線才能對齊。 |
| **另一標的代碼與圖表交易所不符** | 例如圖表是 BYBIT:ETHUSDT 卻手動填 BINANCE:BTCUSDT，4H 時間軸可能對不齊、產生大量 na。改為自動或手動填 BYBIT:BTCUSDT。 |
| **歷史 K 線太少** | 策略需至少約 50+ 根 4H 才可能出現訊號（swing 確認 + 突破 + 16 根監控）。請在圖表上往左拉長歷史或放大週期。 |
| **策略從未觸發進場** | 若仍無交易，屬正常（背離條件較嚴）；圖表應至少會顯示 K 線，不會再顯示 not enough data。 |

腳本已做：自動另一標的（同交易所）、`request.security` 使用 `gaps=barmerge.gaps_off` 填補缺棒、以及錨定 plot 確保有資料可顯示。

## 注意事項

- **Pine 與 Python 回測差異**：Pine 的 swing 用 `ta.pivothigh/pivotlow`，failure 價用區間內最高/最低近似，與 Python 的「最近 swing high/low」可能略有差異，訊號數量與時點不會完全一致。  
- **僅限 4H**：規則與回測皆以 4H 為準，其他週期僅供參考。  
- **策略下單**：TradingView 策略下單只會下在**圖表標的**（ETH），無法在同一腳本內對 BTC 下單。

## 警報

Indicator 版內建兩個 alertcondition：  
- **背離空訊**：Bearish divergence 進場空訊觸發  
- **背離多訊**：Bullish divergence 進場多訊觸發  

可在圖表上右鍵 → 添加警報 → 條件選該指標與對應 alert。
