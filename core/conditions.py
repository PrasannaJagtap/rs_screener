"""
conditions.py
-------------
All 12 screening conditions as individual functions.
Each function takes the RS summary DataFrame and returns True/False.

Groups:
  Group 1 — Mandatory (Classic RS System)    : C1–C5
  Group 2 — Early Recovery (Refined Signal)  : C6–C9
  Group 3 — Quality Filters                  : C10–C12

Scoring:
  Each passing condition = 1 point
  User selects which conditions to check (via Streamlit checkboxes)
  Score = count of selected conditions that pass
  Tier 1 = passes ALL selected conditions
  Tier 2 = passes >= minimum threshold
  Watchlist = one condition below threshold
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_fetcher import fetch_stock, fetch_benchmark, align_data
from core.rs_calculator import build_rs_summary


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 1 — MANDATORY (Classic RS System)
# ─────────────────────────────────────────────────────────────────────────────

def c1_rs_positive(summary: pd.DataFrame) -> bool:
    """
    C1: RS > 0
    Stock is outperforming benchmark over the lookback period.
    This is the primary entry gate — filters ~60% of all stocks.
    """
    latest_rs = summary["RS"].iloc[-1]
    return bool(latest_rs > 0)


def c2_rs_slope_positive(summary: pd.DataFrame) -> bool:
    """
    C2: RS Slope is positive
    RS is currently rising, not fading.
    Prevents buying leaders that are losing momentum.
    """
    latest_slope = summary["RS_Slope"].iloc[-1]
    return bool(latest_slope > 0)


def c3_rs_ma_rising(summary: pd.DataFrame) -> bool:
    """
    C3: RS MA rising for 3+ consecutive bars
    Sustained trend, not a one-week noise spike.
    This is the trend confirmation filter.
    """
    return bool(summary["RS_MA_Rising_3bars"].iloc[-1])


def c4_price_above_sma(summary: pd.DataFrame, sma_col: str = "Price_SMA_20") -> bool:
    """
    C4: Price above SMA AND SMA is rising
    Price trend must confirm RS strength.
    RS can be positive while price is in a downtrend — avoid this.
    """
    close     = summary["Close"].iloc[-1]
    sma_now   = summary[sma_col].iloc[-1]
    sma_prev  = summary[sma_col].iloc[-2]   # one week ago

    price_above_sma = close > sma_now
    sma_rising      = sma_now > sma_prev

    return bool(price_above_sma and sma_rising)


def c5_price_above_wma200(summary: pd.DataFrame) -> bool:
    """
    C5: Price above WMA 200
    Long-term trend filter. Ensures we're buying in an uptrend,
    not catching a falling knife.
    WMA 200 weekly = ~4 year weighted average.
    """
    close   = summary["Close"].iloc[-1]
    wma200  = summary["Price_WMA_200"].iloc[-1]

    if pd.isna(wma200):
        return False   # Not enough data to calculate

    return bool(close > wma200)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 2 — EARLY RECOVERY (Refined Signal)
# ─────────────────────────────────────────────────────────────────────────────

def c6_rs_crossed_above_rs_ma(summary: pd.DataFrame) -> bool:
    """
    C6: RS just crossed above RS MA (both can be negative)
    Early turn signal — catches recoveries before RS crosses zero.
    Requires C7, C8, C9 to filter noise.
    """
    rs    = summary["RS"]
    rs_ma = summary["RS_MA"]

    # Check if RS crossed above RS MA in the last 2 bars
    was_below = rs.iloc[-2] < rs_ma.iloc[-2]
    is_above  = rs.iloc[-1] >= rs_ma.iloc[-1]

    return bool(was_below and is_above)


def c7_rs_depth_limit(summary: pd.DataFrame, depth_limit: float = -0.10) -> bool:
    """
    C7: RS > -0.10 (depth limit)
    Prevents catching deeply negative stocks.
    RS at -0.35 needs 35 weeks of 1% weekly outperformance to recover.
    Only consider stocks close to zero (near-zero negative allowed).
    """
    latest_rs = summary["RS"].iloc[-1]
    return bool(latest_rs > depth_limit)


def c8_former_leader(summary: pd.DataFrame) -> bool:
    """
    C8: Stock was RS positive at some point in last 52 weeks
    Filters chronic underperformers.
    Only former leaders pulling back qualify for early recovery signals.
    Southwest Pinnacle pattern: leader → pullback → re-entry.
    """
    rs_last_52 = summary["RS"].iloc[-52:]
    max_rs = rs_last_52.max()
    return bool(max_rs > 0)


def c9_crossover_sustained(summary: pd.DataFrame) -> bool:
    """
    C9: RS has been above RS MA for at least 2 consecutive bars
    Confirms the crossover is real, not a one-bar noise spike.
    """
    rs    = summary["RS"]
    rs_ma = summary["RS_MA"]

    # Last 2 bars: RS must be above RS MA on both
    bar1_above = rs.iloc[-1] >= rs_ma.iloc[-1]
    bar2_above = rs.iloc[-2] >= rs_ma.iloc[-2]

    return bool(bar1_above and bar2_above)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 3 — QUALITY FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def c10_minimum_volume(
    stock_df: pd.DataFrame,
    min_avg_volume: int = 500_000
) -> bool:
    """
    C10: Average weekly volume > 500,000 shares
    Filters illiquid stocks — SME stocks, suspended stocks, etc.
    Institutions can't enter/exit positions in low-volume stocks.
    Uses 20-week average volume.
    """
    if "Volume" not in stock_df.columns:
        return True   # If volume data unavailable, don't penalize

    avg_vol = stock_df["Volume"].iloc[-20:].mean()

    if pd.isna(avg_vol):
        return True   # Same — no data, don't penalize

    return bool(avg_vol > min_avg_volume)


def c11_near_52w_high(summary: pd.DataFrame, threshold: float = 0.70) -> bool:
    """
    C11: Price within 30% of 52-week high
    Stocks more than 30% below their high are usually in trouble.
    Leaders consolidate near highs — they don't fall 40% and recover easily.
    threshold = 0.70 means price >= 70% of 52-week high.
    """
    close      = summary["Close"].iloc[-1]
    high_52w   = summary["Close"].iloc[-52:].max()

    return bool(close >= high_52w * threshold)


def c12_rs_zero_crossover_recent(summary: pd.DataFrame) -> bool:
    """
    C12: RS crossed above zero within the last 8 bars
    Best entries happen close to the zero crossover.
    Extended RS (50+ bars positive at high levels) = late entry risk.
    This catches fresh breakouts.
    """
    return bool(summary["RS_Crossed_Zero_Recently"].iloc[-1])


# ─────────────────────────────────────────────────────────────────────────────
# MASTER SCORER
# ─────────────────────────────────────────────────────────────────────────────

# Default condition config — which conditions are active by default
DEFAULT_CONDITIONS = {
    "C1_RS_Positive"          : True,
    "C2_RS_Slope_Positive"    : True,
    "C3_RS_MA_Rising"         : True,
    "C4_Price_Above_SMA"      : True,
    "C5_Above_WMA200"         : True,
    "C6_RS_Crossed_Above_MA"  : False,   # Early recovery — off by default
    "C7_RS_Depth_Limit"       : False,
    "C8_Former_Leader"        : False,
    "C9_Crossover_Sustained"  : False,
    "C10_Min_Volume"          : True,
    "C11_Near_52w_High"       : True,
    "C12_RS_Zero_Cross_Recent": False,   # Optional bonus — off by default
}


def score_stock(
    summary:    pd.DataFrame,
    stock_df:   pd.DataFrame,
    conditions: dict = None,
    min_threshold: int = 4,
) -> dict:
    """
    Run all active conditions against a stock and return a score report.

    Parameters
    ----------
    summary       : DataFrame from build_rs_summary()
    stock_df      : Raw stock DataFrame (needed for volume check)
    conditions    : dict of {condition_name: bool} — which to check
                    Defaults to DEFAULT_CONDITIONS if None
    min_threshold : Minimum score to qualify as Tier 2

    Returns
    -------
    dict with individual condition results, score, and tier
    """
    if conditions is None:
        conditions = DEFAULT_CONDITIONS

    # Run each condition
    results = {}

    if conditions.get("C1_RS_Positive"):
        results["C1_RS_Positive"] = c1_rs_positive(summary)

    if conditions.get("C2_RS_Slope_Positive"):
        results["C2_RS_Slope_Positive"] = c2_rs_slope_positive(summary)

    if conditions.get("C3_RS_MA_Rising"):
        results["C3_RS_MA_Rising"] = c3_rs_ma_rising(summary)

    if conditions.get("C4_Price_Above_SMA"):
        results["C4_Price_Above_SMA"] = c4_price_above_sma(summary)

    if conditions.get("C5_Above_WMA200"):
        results["C5_Above_WMA200"] = c5_price_above_wma200(summary)

    if conditions.get("C6_RS_Crossed_Above_MA"):
        results["C6_RS_Crossed_Above_MA"] = c6_rs_crossed_above_rs_ma(summary)

    if conditions.get("C7_RS_Depth_Limit"):
        results["C7_RS_Depth_Limit"] = c7_rs_depth_limit(summary)

    if conditions.get("C8_Former_Leader"):
        results["C8_Former_Leader"] = c8_former_leader(summary)

    if conditions.get("C9_Crossover_Sustained"):
        results["C9_Crossover_Sustained"] = c9_crossover_sustained(summary)

    if conditions.get("C10_Min_Volume"):
        results["C10_Min_Volume"] = c10_minimum_volume(stock_df)

    if conditions.get("C11_Near_52w_High"):
        results["C11_Near_52w_High"] = c11_near_52w_high(summary)

    if conditions.get("C12_RS_Zero_Cross_Recent"):
        results["C12_RS_Zero_Cross_Recent"] = c12_rs_zero_crossover_recent(summary)

    # Score
    total_active = len(results)
    score        = sum(1 for v in results.values() if v)

    # Tier
    if score == total_active:
        tier = "Tier 1"
    elif score >= min_threshold:
        tier = "Tier 2"
    elif score == min_threshold - 1:
        tier = "Watchlist"
    else:
        tier = "Filtered"

    return {
        "conditions"   : results,
        "score"        : score,
        "total_active" : total_active,
        "min_threshold": min_threshold,
        "tier"         : tier,
    }


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    TEST_STOCKS = [
        "RELIANCE",
        "HDFCBANK",
        "INFY",
        "TATAMOTORS",
        "WIPRO",
    ]

    print("=" * 65)
    print("  CONDITIONS — VERIFICATION TEST")
    print("  Testing 5 stocks against all 12 conditions (Weekly)")
    print("=" * 65)

    bench = fetch_benchmark("^CNX200")
    if bench is None:
        print("❌ Could not fetch benchmark. Exiting.")
        exit()

    # Run all conditions with defaults (C1-C5, C10, C11 active)
    results_table = []

    for symbol in TEST_STOCKS:
        print(f"\n📊 Processing {symbol}...")

        stock_df = fetch_stock(symbol)
        if stock_df is None:
            print(f"  ⚠️  Skipping {symbol} — no data")
            continue

        s_aligned, b_aligned = align_data(stock_df, bench)

        summary = build_rs_summary(
            stock_close  = s_aligned["Close"],
            bench_close  = b_aligned["Close"],
            rs_period    = 52,
            ma_period    = 20,
            sma_period   = 20,
            slope_period = 5,
        )

        report = score_stock(
            summary       = summary,
            stock_df      = stock_df,
            conditions    = DEFAULT_CONDITIONS,
            min_threshold = 4,
        )

        # Quick condition display
        latest = summary.iloc[-1]
        conds  = report["conditions"]

        print(f"  Close : ₹{latest['Close']:,.2f}")
        print(f"  RS    : {latest['RS']:+.4f}  |  RS_MA: {latest['RS_MA']:.4f}")
        print(f"  Score : {report['score']}/{report['total_active']}  →  {report['tier']}")
        print(f"  ┌─ C1 RS>0         : {'✅' if conds.get('C1_RS_Positive') else '❌'}")
        print(f"  ├─ C2 RS Slope ↑   : {'✅' if conds.get('C2_RS_Slope_Positive') else '❌'}")
        print(f"  ├─ C3 RS MA Rising : {'✅' if conds.get('C3_RS_MA_Rising') else '❌'}")
        print(f"  ├─ C4 Price>SMA    : {'✅' if conds.get('C4_Price_Above_SMA') else '❌'}")
        print(f"  ├─ C5 Above WMA200 : {'✅' if conds.get('C5_Above_WMA200') else '❌'}")
        print(f"  ├─ C10 Volume      : {'✅' if conds.get('C10_Min_Volume') else '❌'}")
        print(f"  └─ C11 Near 52wHigh: {'✅' if conds.get('C11_Near_52w_High') else '❌'}")

        results_table.append({
            "Symbol": symbol,
            "Close" : round(latest["Close"], 2),
            "RS"    : round(latest["RS"], 4),
            "Score" : f"{report['score']}/{report['total_active']}",
            "Tier"  : report["tier"],
        })

    # Summary table
    print("\n" + "=" * 65)
    print("  SUMMARY TABLE")
    print("=" * 65)
    print(f"  {'Symbol':<14} {'Close':>10}  {'RS':>8}  {'Score':>6}  {'Tier'}")
    print(f"  {'-'*14} {'-'*10}  {'-'*8}  {'-'*6}  {'-'*10}")
    for r in results_table:
        rs_icon = "🟢" if r["RS"] > 0 else "🔴"
        print(f"  {r['Symbol']:<14} ₹{r['Close']:>9,.2f}  {rs_icon}{r['RS']:>+7.4f}  {r['Score']:>6}  {r['Tier']}")

    print("\n✅ Conditions engine working correctly.")
    print("   Next step: screener.py — loop through all NSE stocks")
