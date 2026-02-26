# 聊天存檔索引

| 日期 | 檔名 | 摘要 | 涉及參數 |
|-----|------|------|--------|
| 2026-02-26 | 2026-02-26-fetch-and-backtest.md | 首次執行 fetch_data → detect_divergence → divergenceplus 完整回測；修正 Python 3.9 相容與 K 線 index 解析 | fetch_data, load_ohlcv, Optional\[dict\], 313 筆、勝率 67.7%、期望值 +2.03% |
| 2026-02-26 | 2026-02-26-integration-to-kline-research.md | **【規範化集成】** 代碼兼容性分析 + 三方案評估 + 方案 A 選定 + 目錄結構與文檔規劃 | 方案 A（保留獨立）、MODELING_RULES RULE-1~10 驗證、README/config/COMPLIANCE.md 規劃 |
| 2026-02-26 | 2026-02-26-divergence-integration.md | divergence-detector → kline-research/projects 集成規劃與執行 | swing_lookback=3, max_divergence_days=5, eth_lookback_days=5, rr_ratio=2.0, time_stop_bars=20 |
| 2026-02-26 | 2026-02-26-btc-backtest-asset-param.md | divergenceplus.py 新增 `--asset BTC\|ETH`，支援獨立 BTC/ETH 回測，輸出各自 trades.csv 和 equity.csv | --asset, _ASSET_FIELDS, btc_hh_price, btc_ll_price, btc_failure_price |
