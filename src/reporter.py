"""
報告輸出模組
統一格式輸出：勝率、盈虧比、期望值、年份分拆、失敗分析
符合 RULE-6 [V4] 驗證報告要求
"""
import os
import pandas as pd


MARKET_LABELS = {
    2018: "熊市",
    2019: "震盪→牛",
    2020: "牛市",
    2021: "強牛→熊轉",
    2022: "熊市",
    2023: "復甦",
    2024: "牛市",
    2025: "牛市/未定",
}


def print_report(trades_df: pd.DataFrame, df: pd.DataFrame, cfg: dict) -> None:
    """輸出完整回測報告。"""

    comp = trades_df.dropna(subset=["exit_time"]).copy()
    if comp.empty:
        print("❌ 沒有完成的交易記錄")
        return

    total = len(comp)
    wins = comp[comp["win"] == True]
    losses = comp[comp["win"] == False]
    wr = len(wins) / total * 100
    avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0
    expectancy = (wr / 100 * avg_win) + ((1 - wr / 100) * avg_loss)
    pf = (wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum())
          if losses["pnl_pct"].sum() != 0 else float("inf"))
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    sym = cfg["data"]["symbol"]
    ivl = cfg["data"]["interval"]
    strat = cfg.get("strategy_name", "unknown")
    ver = cfg.get("strategy_version", "v1")

    DIV = "═" * 64
    div = "─" * 64

    print(f"\n{DIV}")
    print(f"    📊  {sym} {ivl} | {strat} {ver}")
    print(f"    歷史回測報告")
    print(DIV)

    print(f"\n  📅 回測期間    {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  🕯️  K棒總數    {len(df):,} 根 {ivl} K棒")
    print(f"  🔔 完成交易    {total} 筆\n")

    print(f"  {div}")
    print(f"  ─── 核心績效")
    print(f"  {div}")
    print(f"  🏆 勝率          {wr:.1f}%   ({len(wins)} 勝 / {len(losses)} 敗)")
    print(f"  📈 平均盈利      +{avg_win:.2f}%")
    print(f"  📉 平均虧損      {avg_loss:.2f}%")
    print(f"  ⚖️  盈虧比        {rr:.2f}x")
    print(f"  💡 每筆期望值    {expectancy:+.3f}%")
    print(f"  🔥 獲利因子      {pf:.2f}")
    avg_bars = comp["bars_held"].mean()
    avg_hours = comp["hours_held"].mean() if "hours_held" in comp.columns else avg_bars * 4
    print(f"  ⏱️  平均持倉      {avg_bars:.1f} 根棒 ({avg_hours:.0f} 小時)\n")

    print(f"  {div}")
    print(f"  ─── 出場原因分布")
    print(f"  {div}")
    exit_info = {
        "take_profit": ("🟢", "止盈 (RSI 達標)   "),
        "stop_loss":   ("🔴", "止損 (形態失效)   "),
        "time_stop":   ("🟡", "時間止損          "),
    }
    for reason, (em, lb) in exit_info.items():
        sub = comp[comp["exit_reason"] == reason]
        if sub.empty:
            continue
        avg_p = sub["pnl_pct"].mean()
        print(f"  {em} {lb} {len(sub):>3} 筆 ({len(sub)/total*100:.1f}%)  均報酬 {avg_p:+.2f}%")

    print(f"\n  {div}")
    print(f"  ─── 按年份分拆（RULE-6 [V4] 要求）")
    print(f"  {div}")
    print(f"  {'年份':<6} {'筆數':>4} {'勝率':>7} {'均盈':>8} {'均損':>8} {'期望值':>9}  市況")
    print(f"  {'─'*60}")
    for yr in sorted(comp["year"].unique()):
        yt = comp[comp["year"] == yr]
        yw = yt[yt["win"] == True]
        yl = yt[yt["win"] == False]
        y_wr = len(yw) / len(yt) * 100
        y_aw = yw["pnl_pct"].mean() if len(yw) > 0 else 0
        y_al = yl["pnl_pct"].mean() if len(yl) > 0 else 0
        y_ex = (y_wr / 100 * y_aw) + ((1 - y_wr / 100) * y_al)
        ml = MARKET_LABELS.get(yr, "")
        print(f"  {yr:<6} {len(yt):>4} {y_wr:>6.1f}% {y_aw:>+7.2f}% "
              f"{y_al:>+7.2f}% {y_ex:>+8.3f}%  {ml}")

    print(f"\n  {div}")
    print(f"  ─── 失敗案例深度分析")
    print(f"  {div}")

    sl_t = comp[comp["exit_reason"] == "stop_loss"]
    ts_t = comp[comp["exit_reason"] == "time_stop"]

    if not sl_t.empty:
        print(f"\n  🔴 止損 ({len(sl_t)} 筆)")
        if "sl_pct" in sl_t.columns:
            print(f"     → 平均止損距離 : {sl_t['sl_pct'].mean():.2f}%")
        print(f"     → 平均虧損     : {sl_t['pnl_pct'].mean():.2f}%")
        print(f"     → 最大單筆虧損 : {sl_t['pnl_pct'].min():.2f}%")
        by_yr = sl_t.groupby("year").size().to_dict()
        print(f"     → 年份分布     : {by_yr}")

    if not ts_t.empty:
        ts_w = ts_t[ts_t["pnl_pct"] > 0]
        ts_l = ts_t[ts_t["pnl_pct"] <= 0]
        print(f"\n  🟡 時間止損 ({len(ts_t)} 筆)")
        print(f"     → 正報酬 : {len(ts_w)} 筆  均 {ts_w['pnl_pct'].mean():+.2f}%"
              if len(ts_w) > 0 else "     → 正報酬 : 0 筆")
        print(f"     → 負報酬 : {len(ts_l)} 筆  均 {ts_l['pnl_pct'].mean():+.2f}%"
              if len(ts_l) > 0 else "     → 負報酬 : 0 筆")

    # ── Best / Worst ───────────────────────────
    print(f"\n  {div}")
    print(f"  ─── 最佳 5 筆 vs 最差 5 筆")
    print(f"  {div}")

    def _row(r):
        return (f"  {str(r['entry_time'])[:16]} → {str(r['exit_time'])[:16]}  "
                f"{r['exit_reason']:<12} {r['bars_held']:>3}棒  {r['pnl_pct']:>+7.2f}%")

    print("\n  🏅 最佳 5 筆：")
    for _, r in comp.nlargest(5, "pnl_pct").iterrows():
        print(_row(r))

    print("\n  💀 最差 5 筆：")
    for _, r in comp.nsmallest(5, "pnl_pct").iterrows():
        print(_row(r))

    print(f"\n  {div}")
    print("  ⚠️  風險提示")
    print(f"  {div}")
    warnings = [
        "歷史回測，未來績效不保證重現",
        f"已含手續費 {cfg['costs']['fee_pct']}% + 滑點 {cfg['costs']['slippage_pct']}%",
        "底背離偵測延遲 swing_window 根棒確認（防未來洩漏）",
        "本回測未含槓桿；實倉槓桿會放大所有虧損",
        "策略通過回測前請先完成 RULE-9 模擬盤驗證",
    ]
    for i, w in enumerate(warnings, 1):
        print(f"  {i}. {w}")

    print(f"\n{DIV}\n")


def save_results(trades_df: pd.DataFrame, cfg: dict) -> str:
    """儲存交易記錄為 CSV 與 XLSX。"""
    out_dir = cfg.get("output", {}).get("results_dir", "results/")
    os.makedirs(out_dir, exist_ok=True)

    base = (f"{cfg['data']['symbol']}_{cfg['data']['interval']}_"
            f"{cfg.get('strategy_name', 'strategy')}_"
            f"{cfg.get('strategy_version', 'v1')}")

    path_csv = os.path.join(out_dir, base + ".csv")
    trades_df.to_csv(path_csv, index=False)

    path_xlsx = os.path.join(out_dir, base + ".xlsx")
    try:
        trades_df.to_excel(path_xlsx, index=False, engine="openpyxl")
        print(f"💾 交易記錄已儲存：{path_csv}、{path_xlsx}")
    except Exception as e:
        print(f"💾 交易記錄已儲存：{path_csv}")
        print(f"   （xlsx 未寫入：{e}）")

    return path_csv
