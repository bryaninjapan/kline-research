# divergenceplus.py 新增 BTC 回測支援

**日期**：2026-02-26
**任務**：為 `divergenceplus.py` 加入 `--asset BTC|ETH` 參數，使 BTC 和 ETH 可獨立回測，產生各自的 trades.csv 和 equity.csv

---

## 問題背景

原始 `divergenceplus.py` 只回測 ETH：
- 硬編碼讀取 `eth_*` JSON 欄位（`eth_rejection_time`, `eth_failure_price`, `eth_neckline`, `eth_high_price`, `eth_low_price`）
- 硬編碼載入 `ETHUSDT_4h.csv`
- 輸出固定命名：`divergenceplus_trades.csv`, `divergenceplus_equity.csv`

但 `detect_divergence.py` 的 JSON 輸出早已同時包含 **BTC 和 ETH 兩組欄位**，只需讀取 `btc_*` 欄位即可跑 BTC 回測。

---

## ETH vs BTC 欄位對照

| 用途 | ETH 欄位 | BTC 欄位 |
|------|----------|----------|
| rejection_time | `eth_rejection_time` | `btc_rejection_time` |
| failure_price | `eth_failure_price` | `btc_failure_price` |
| neckline | `eth_neckline` | `btc_neckline` |
| extreme (bearish H2) | `eth_high_price` | `btc_hh_price` |
| extreme (bullish L2) | `eth_low_price` | `btc_ll_price` |

注意：BTC bearish extreme 叫 `btc_hh_price`（Higher High），ETH 叫 `eth_high_price`（Lower High）— 名稱語意不同但都是第二個 swing 點。

---

## 修改內容（僅 `scripts/divergenceplus.py`）

### 1. 新增 `_ASSET_FIELDS` dict

在 `OUTPUT_DIR` 定義之後加入欄位映射：

```python
_ASSET_FIELDS = {
    "ETH": {
        "rejection_time":  "eth_rejection_time",
        "failure_price":   "eth_failure_price",
        "neckline":        "eth_neckline",
        "extreme_bearish": "eth_high_price",
        "extreme_bullish": "eth_low_price",
    },
    "BTC": {
        "rejection_time":  "btc_rejection_time",
        "failure_price":   "btc_failure_price",
        "neckline":        "btc_neckline",
        "extreme_bearish": "btc_hh_price",
        "extreme_bullish": "btc_ll_price",
    },
}
```

### 2. `_parse_event(ev)` → `_parse_event(ev, asset)`

- 加入 `asset` 參數，查 `_ASSET_FIELDS[asset]` 取欄位名
- return dict 的 key 改為泛用名（去 `eth_` 前綴）：`failure`, `neckline`, `extreme`

### 3. `run_backtest()` — 去除 ETH 綁定

- 簽名加 `asset` 參數：`run_backtest(events, ohlcv_df, cfg, asset)`
- `eth_df` → `ohlcv_df`，`eth_close/high/low/times` → `close/high/low/times`
- `trades.append()` 欄位名：`eth_failure_price` → `failure_price`，`eth_neckline` → `neckline`，`eth_extreme_price` → `extreme_price`

### 4. `parse_args()` — 新增 CLI 引數

- `--asset ETH|BTC`（預設 `ETH`）
- `--eth-csv` 重命名為 `--csv`（泛用）

### 5. `main()` — 動態 symbol 與輸出路徑

```python
asset  = args.asset.upper()   # "ETH" or "BTC"
symbol = f"{asset}USDT"       # "ETHUSDT" or "BTCUSDT"
tag    = asset.lower()        # "eth" or "btc"
```

輸出檔名加 tag 中綴：
- `divergenceplus_{tag}_trades.csv`
- `divergenceplus_{tag}_equity.csv`
- `divergenceplus_{tag}_report.txt`

---

## 使用方式

```bash
# ETH 回測（預設，向下相容）
python scripts/divergenceplus.py

# BTC 回測
python scripts/divergenceplus.py --asset BTC

# 輸出結果
# output/divergenceplus_eth_trades.csv
# output/divergenceplus_eth_equity.csv
# output/divergenceplus_btc_trades.csv
# output/divergenceplus_btc_equity.csv
```

---

## 設計決策

| 選項 | 決策 | 理由 |
|------|------|------|
| 新增 `--asset` 參數 vs 建立獨立 `divergenceplus_btc.py` | `--asset` 參數 | 避免代碼重複，單一維護點 |
| 輸出命名策略 | `_{tag}_` 中綴 | ETH 和 BTC 結果共存於 output/，不互相覆蓋 |
| `--eth-csv` 重命名 | 改為 `--csv` | 泛用化，不限定資產 |
| return dict 欄位名 | 去掉 `eth_` 前綴 | 泛用化，`failure`/`neckline`/`extreme` 在 BTC 語境下也正確 |
