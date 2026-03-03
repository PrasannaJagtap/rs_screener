"""
data_fetcher.py
---------------
Fetches weekly OHLCV price data from yfinance for:
  - NSE stocks (e.g. "RELIANCE.NS")
  - Benchmark indices (e.g. "^CNX200")

All data returned as pandas DataFrames with a clean DatetimeIndex.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


# ── Constants ──────────────────────────────────────────────────────────────────

# How many weeks of data we need:
#   52 (RS lookback) + 20 (MA warmup) + 130 (buffer) = ~200 weeks ≈ 4 years
WEEKS_REQUIRED = 200
DAYS_REQUIRED  = WEEKS_REQUIRED * 7          # yfinance uses calendar days

# Default benchmark
DEFAULT_BENCHMARK = "^CNX200"


# ── Core fetch function ────────────────────────────────────────────────────────

def fetch_weekly_data(ticker: str, weeks: int = WEEKS_REQUIRED) -> pd.DataFrame | None:
    """
    Fetch weekly OHLCV data for a given ticker.

    Parameters
    ----------
    ticker : str
        yfinance ticker symbol.
        NSE stocks  → append .NS  e.g. "RELIANCE.NS"
        Benchmarks  → use ^ prefix e.g. "^CNX200"
    weeks : int
        Number of weeks of history to fetch (default 200).

    Returns
    -------
    pd.DataFrame with columns [Open, High, Low, Close, Volume]
    and a DatetimeIndex (weekly, Friday close for NSE).
    Returns None if data is unavailable or too short.
    """
    start_date = datetime.today() - timedelta(days=weeks * 7)
    start_str  = start_date.strftime("%Y-%m-%d")

    try:
        raw = yf.download(
            tickers   = ticker,
            start     = start_str,
            interval  = "1wk",
            auto_adjust = True,      # adjusts for splits & bonuses automatically
            progress  = False,       # suppress download progress bar
        )
    except Exception as e:
        print(f"  ❌ Download error for {ticker}: {e}")
        return None

    # yfinance sometimes returns MultiIndex columns — flatten them
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Drop rows where Close is NaN
    raw = raw.dropna(subset=["Close"])

    if raw.empty:
        print(f"  ⚠️  No data returned for {ticker}")
        return None

    # Keep only the columns we need
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    df   = raw[cols].copy()

    # Minimum data check — need at least 60 weeks to be useful
    if len(df) < 60:
        print(f"  ⚠️  {ticker} has only {len(df)} weeks — skipping (need ≥ 60)")
        return None

    return df


# ── Convenience wrappers ───────────────────────────────────────────────────────

def fetch_benchmark(symbol: str = DEFAULT_BENCHMARK) -> pd.DataFrame | None:
    """Fetch weekly data for a benchmark index."""
    return fetch_weekly_data(symbol)


def fetch_stock(symbol: str) -> pd.DataFrame | None:
    """
    Fetch weekly data for an NSE stock.
    Automatically appends .NS if not already present.
    """
    if not symbol.endswith(".NS"):
        symbol = symbol + ".NS"
    return fetch_weekly_data(symbol)


# ── Alignment helper ───────────────────────────────────────────────────────────

def align_data(stock_df: pd.DataFrame, bench_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align stock and benchmark DataFrames to the same date index.
    Both DataFrames are trimmed to their common date range.

    Returns
    -------
    (aligned_stock, aligned_benchmark) — both with identical DatetimeIndex
    """
    common_dates = stock_df.index.intersection(bench_df.index)
    return stock_df.loc[common_dates], bench_df.loc[common_dates]


# ── Quick verification (run this file directly to test) ───────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("  DATA FETCHER — VERIFICATION TEST")
    print("=" * 55)

    # ── Test 1: Benchmark ──────────────────────────────────────
    print("\n📊 Fetching CNX200 benchmark...")
    bench = fetch_benchmark("^CNX200")

    if bench is not None:
        print(f"  ✅ CNX200 loaded")
        print(f"     Rows       : {len(bench)} weeks")
        print(f"     Date range : {bench.index[0].date()} → {bench.index[-1].date()}")
        print(f"     Columns    : {list(bench.columns)}")
        print(f"     Latest close: ₹{bench['Close'].iloc[-1]:,.2f}")
    else:
        print("  ❌ CNX200 failed — check internet / ticker symbol")

    # ── Test 2: Reliance ───────────────────────────────────────
    print("\n📈 Fetching RELIANCE.NS (test stock)...")
    reliance = fetch_stock("RELIANCE")

    if reliance is not None:
        print(f"  ✅ RELIANCE loaded")
        print(f"     Rows       : {len(reliance)} weeks")
        print(f"     Date range : {reliance.index[0].date()} → {reliance.index[-1].date()}")
        print(f"     Latest close: ₹{reliance['Close'].iloc[-1]:,.2f}")
    else:
        print("  ❌ RELIANCE failed")

    # ── Test 3: Alignment ──────────────────────────────────────
    if bench is not None and reliance is not None:
        print("\n🔗 Testing data alignment...")
        rel_aligned, bench_aligned = align_data(reliance, bench)
        print(f"  ✅ Aligned rows : {len(rel_aligned)} weeks")
        print(f"     Stock rows   : {len(reliance)} → {len(rel_aligned)} after alignment")
        print(f"     Bench rows   : {len(bench)}    → {len(bench_aligned)} after alignment")
        dates_match = rel_aligned.index.equals(bench_aligned.index)
        print(f"     Dates match  : {'✅ Yes' if dates_match else '❌ No'}")

    # ── Test 4: Bad ticker ─────────────────────────────────────
    print("\n🚫 Testing bad ticker (should fail gracefully)...")
    bad = fetch_stock("XYZXYZXYZ_INVALID")
    print(f"  Result: {'None returned ✅' if bad is None else 'Got data (unexpected)'}")

    print("\n" + "=" * 55)
    print("  TEST COMPLETE")
    print("=" * 55)
