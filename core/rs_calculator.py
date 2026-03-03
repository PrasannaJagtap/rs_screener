"""
rs_calculator.py
----------------
Calculates Relative Strength (RS) indicators for a stock vs a benchmark.

Formula used (matches TradingView's RS implementation):
    RS = (Stock_today / Stock_N_bars_ago) / (Bench_today / Bench_N_bars_ago) - 1

Outputs:
    rs_line     : Raw RS value each week
    rs_ma       : Simple Moving Average of RS line
    rs_slope    : Direction of RS (positive = rising, negative = falling)
    rs_ma_slope : Direction of RS MA (3 consecutive bars rising = confirmed)
    price_sma   : Simple Moving Average of stock price
    price_wma200: Weighted Moving Average of stock price (200 periods)
"""

import numpy as np
import pandas as pd
import sys
import os

# So we can import from sibling folder when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_fetcher import fetch_stock, fetch_benchmark, align_data


# ── Core RS Calculation ────────────────────────────────────────────────────────

def calculate_rs(
    stock_close:  pd.Series,
    bench_close:  pd.Series,
    rs_period:    int = 52,
) -> pd.Series:
    """
    Calculate the RS line.

    RS = (Stock_today / Stock_N_bars_ago) / (Bench_today / Bench_N_bars_ago) - 1

    Parameters
    ----------
    stock_close : pd.Series  — weekly close prices of the stock
    bench_close : pd.Series  — weekly close prices of the benchmark
    rs_period   : int        — lookback period in bars (52 = 1 year weekly)

    Returns
    -------
    pd.Series of RS values (NaN for first rs_period bars)
    """
    # Performance over the lookback window
    stock_perf = stock_close / stock_close.shift(rs_period)
    bench_perf = bench_close / bench_close.shift(rs_period)

    # RS = relative outperformance
    rs = (stock_perf / bench_perf) - 1

    rs.name = "RS"
    return rs


def calculate_rs_ma(rs_line: pd.Series, ma_period: int = 20) -> pd.Series:
    """
    Calculate the Simple Moving Average of the RS line.

    Parameters
    ----------
    rs_line  : pd.Series — the RS values
    ma_period: int       — smoothing period (20 bars for weekly)

    Returns
    -------
    pd.Series of RS MA values
    """
    rs_ma = rs_line.rolling(window=ma_period).mean()
    rs_ma.name = "RS_MA"
    return rs_ma


def calculate_rs_slope(rs_line: pd.Series, slope_period: int = 5) -> pd.Series:
    """
    Calculate whether the RS line is rising or falling.

    slope_period bars ago → today: positive = rising, negative = falling.

    Parameters
    ----------
    rs_line      : pd.Series
    slope_period : int — how many bars back to compare (5 = ~1 month weekly)

    Returns
    -------
    pd.Series of floats: positive = RS rising, negative = RS falling
    """
    slope = rs_line - rs_line.shift(slope_period)
    slope.name = "RS_Slope"
    return slope


def calculate_price_sma(close: pd.Series, sma_period: int = 20) -> pd.Series:
    """Simple Moving Average of price."""
    sma = close.rolling(window=sma_period).mean()
    sma.name = f"SMA_{sma_period}"
    return sma


def calculate_price_wma(close: pd.Series, wma_period: int = 200) -> pd.Series:
    """
    Weighted Moving Average of price.
    More recent bars get higher weight.
    WMA(200) is the long-term trend filter.
    """
    weights = np.arange(1, wma_period + 1, dtype=float)

    def wma(x):
        if len(x) < wma_period:
            return np.nan
        return np.dot(x, weights) / weights.sum()

    wma_series = close.rolling(window=wma_period).apply(wma, raw=True)
    wma_series.name = f"WMA_{wma_period}"
    return wma_series


# ── RS MA Rising Check ─────────────────────────────────────────────────────────

def is_rs_ma_rising(rs_ma: pd.Series, consecutive: int = 3) -> pd.Series:
    """
    Check if RS MA has been rising for N consecutive bars.

    This is Checkpoint 3: sustained RS MA trend confirmation.

    Returns
    -------
    pd.Series of bool — True if RS MA rose for `consecutive` bars in a row
    """
    rising = rs_ma > rs_ma.shift(1)

    result = pd.Series(False, index=rs_ma.index)
    for i in range(consecutive - 1, len(rs_ma)):
        if all(rising.iloc[i - j] for j in range(consecutive)):
            result.iloc[i] = True

    result.name = "RS_MA_Rising"
    return result


def rs_crossed_zero_recently(rs_line: pd.Series, within_bars: int = 8) -> pd.Series:
    """
    Check if RS crossed above zero within the last N bars.

    This is Checkpoint 5 (bonus): catching early stage leaders.
    Best entries are within 8 bars of RS crossing zero.

    Returns
    -------
    pd.Series of bool
    """
    was_negative = rs_line.shift(1) < 0
    is_positive  = rs_line >= 0
    crossed      = was_negative & is_positive

    # Look back within_bars to see if a crossover happened recently
    result = crossed.rolling(window=within_bars, min_periods=1).max().astype(bool)
    result.name = "RS_Crossed_Zero_Recently"
    return result


def rs_crossed_above_rs_ma_recently(
    rs_line: pd.Series,
    rs_ma:   pd.Series,
    within_bars: int = 2
) -> pd.Series:
    """
    Check if RS crossed above RS MA recently (within N bars).

    Used for the Early Recovery signal (Checkpoint 6).

    Returns
    -------
    pd.Series of bool
    """
    was_below = rs_line.shift(1) < rs_ma.shift(1)
    is_above  = rs_line >= rs_ma
    crossed   = was_below & is_above

    result = crossed.rolling(window=within_bars, min_periods=1).max().astype(bool)
    result.name = "RS_Crossed_Above_MA"
    return result


# ── Full Summary Builder ───────────────────────────────────────────────────────

def build_rs_summary(
    stock_close: pd.Series,
    bench_close: pd.Series,
    rs_period:   int = 52,
    ma_period:   int = 20,
    sma_period:  int = 20,
    slope_period:int = 5,
) -> pd.DataFrame:
    """
    Build a complete RS summary DataFrame for a stock vs benchmark.

    Returns a DataFrame with all RS indicators as columns.
    """
    rs        = calculate_rs(stock_close, bench_close, rs_period)
    rs_ma     = calculate_rs_ma(rs, ma_period)
    rs_slope  = calculate_rs_slope(rs, slope_period)
    price_sma = calculate_price_sma(stock_close, sma_period)
    price_wma = calculate_price_wma(stock_close, 200)
    rs_ma_rising = is_rs_ma_rising(rs_ma, consecutive=3)
    rs_zero_cross = rs_crossed_zero_recently(rs, within_bars=8)

    df = pd.DataFrame({
        "Close"                  : stock_close,
        "Bench_Close"            : bench_close,
        "RS"                     : rs,
        "RS_MA"                  : rs_ma,
        "RS_Slope"               : rs_slope,
        f"Price_SMA_{sma_period}": price_sma,
        "Price_WMA_200"          : price_wma,
        "RS_MA_Rising_3bars"     : rs_ma_rising,
        "RS_Crossed_Zero_Recently": rs_zero_cross,
    })

    return df


# ── Verification Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  RS CALCULATOR — VERIFICATION TEST")
    print("  Stock: RELIANCE vs Benchmark: CNX200 (Weekly, 52-bar)")
    print("=" * 60)

    # Fetch data
    print("\n📥 Fetching data...")
    bench    = fetch_benchmark("^CNX200")
    reliance = fetch_stock("RELIANCE")

    if bench is None or reliance is None:
        print("❌ Could not fetch data. Run data_fetcher.py first.")
        exit()

    # Align
    rel_aligned, bench_aligned = align_data(reliance, bench)
    print(f"  ✅ Aligned: {len(rel_aligned)} weeks")

    # Build RS summary
    print("\n⚙️  Calculating RS indicators...")
    summary = build_rs_summary(
        stock_close  = rel_aligned["Close"],
        bench_close  = bench_aligned["Close"],
        rs_period    = 52,
        ma_period    = 20,
        sma_period   = 20,
        slope_period = 5,
    )

    # ── Print last 8 weeks ─────────────────────────────────────
    print("\n📋 LAST 8 WEEKS OF DATA:")
    print("-" * 60)

    last8 = summary.tail(8)[[
        "Close", "RS", "RS_MA", "RS_Slope",
        "Price_SMA_20", "RS_MA_Rising_3bars"
    ]].copy()

    # Format for readability
    last8["Close"]      = last8["Close"].round(2)
    last8["RS"]         = last8["RS"].round(4)
    last8["RS_MA"]      = last8["RS_MA"].round(4)
    last8["RS_Slope"]   = last8["RS_Slope"].round(4)
    last8["Price_SMA_20"] = last8["Price_SMA_20"].round(2)

    # Print row by row
    for date, row in last8.iterrows():
        rs_color  = "🟢" if row["RS"] > 0 else "🔴"
        slp_color = "↑" if row["RS_Slope"] > 0 else "↓"
        ma_up     = "✅" if row["RS_MA_Rising_3bars"] else "  "
        print(
            f"  {str(date.date())}"
            f"  Close: ₹{row['Close']:>8.2f}"
            f"  RS: {rs_color}{row['RS']:>7.4f}"
            f"  RS_MA: {row['RS_MA']:>7.4f}"
            f"  Slope: {slp_color}{abs(row['RS_Slope']):.4f}"
            f"  MA_Rising: {ma_up}"
        )

    # ── Current snapshot ──────────────────────────────────────
    latest = summary.iloc[-1]
    print("\n" + "=" * 60)
    print("  CURRENT SNAPSHOT (Latest Week)")
    print("=" * 60)
    print(f"  Stock Close   : ₹{latest['Close']:.2f}")
    print(f"  Bench Close   : ₹{latest['Bench_Close']:.2f}")
    print(f"  RS Value      : {latest['RS']:.4f}  {'(outperforming ✅)' if latest['RS'] > 0 else '(underperforming ❌)'}")
    print(f"  RS MA         : {latest['RS_MA']:.4f}")
    print(f"  RS vs RS_MA   : {'RS above MA ✅' if latest['RS'] > latest['RS_MA'] else 'RS below MA ❌'}")
    print(f"  RS Slope      : {'Positive ↑ ✅' if latest['RS_Slope'] > 0 else 'Negative ↓ ❌'}")
    print(f"  Price SMA 20  : ₹{latest['Price_SMA_20']:.2f}")
    print(f"  Price WMA 200 : ₹{latest['Price_WMA_200']:.2f}")
    above_sma = latest['Close'] > latest['Price_SMA_20']
    above_wma = latest['Close'] > latest['Price_WMA_200']
    print(f"  Above SMA 20  : {'✅ Yes' if above_sma else '❌ No'}")
    print(f"  Above WMA 200 : {'✅ Yes' if above_wma else '❌ No'}")
    print(f"  RS MA Rising  : {'✅ Yes (3 consecutive bars)' if latest['RS_MA_Rising_3bars'] else '❌ No'}")
    print(f"  RS Near Zero  : {'✅ Crossed zero recently (<8 bars)' if latest['RS_Crossed_Zero_Recently'] else 'Not recent'}")

    print("\n" + "=" * 60)
    print("  NOW: Compare RS Value above against TradingView")
    print("  TradingView → RELIANCE weekly chart")
    print("  Add indicator: 'Relative Strength' (not RSI!)")
    print("  Settings: Source=CNX200, Period=52, MA=20")
    print("  The RS value should be close to what's shown above.")
    print("=" * 60)
