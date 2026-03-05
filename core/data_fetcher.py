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
import time
from datetime import datetime, timedelta


# ── Constants ──────────────────────────────────────────────────────────────────

WEEKS_REQUIRED    = 200
DAYS_REQUIRED     = WEEKS_REQUIRED * 7
DEFAULT_BENCHMARK = "^CNX200"


# ── Core fetch function ────────────────────────────────────────────────────────

def fetch_weekly_data(ticker: str, weeks: int = WEEKS_REQUIRED) -> pd.DataFrame | None:
    """
    Fetch weekly OHLCV data for a given ticker.
    Retries up to 3 times with backoff on rate limit errors.
    """
    start_date = datetime.today() - timedelta(days=weeks * 7)
    start_str  = start_date.strftime("%Y-%m-%d")

    max_retries  = 3
    retry_delays = [30, 60, 120]

    raw = None
    for attempt in range(max_retries):
        try:
            raw = yf.download(
                tickers     = ticker,
                start       = start_str,
                interval    = "1wk",
                auto_adjust = True,
                progress    = False,
            )
            break

        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "too many requests" in err_str or "429" in err_str:
                if attempt < max_retries - 1:
                    wait = retry_delays[attempt]
                    print(f"\n  ⏳ Rate limited on {ticker}. "
                          f"Waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait)
                    continue
                else:
                    print(f"  ❌ Rate limit persists for {ticker} after {max_retries} attempts")
                    return None
            else:
                print(f"  ❌ Download error for {ticker}: {e}")
                return None

    if raw is None or raw.empty:
        print(f"  ⚠️  No data returned for {ticker}")
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.dropna(subset=["Close"])

    if raw.empty:
        print(f"  ⚠️  No data returned for {ticker}")
        return None

    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    df   = raw[cols].copy()

    if len(df) < 60:
        print(f"  ⚠️  {ticker} has only {len(df)} weeks — skipping (need ≥ 60)")
        return None

    return df


# ── Convenience wrappers ───────────────────────────────────────────────────────

def fetch_benchmark(symbol: str = DEFAULT_BENCHMARK) -> pd.DataFrame | None:
    """Fetch weekly data for a benchmark index."""
    return fetch_weekly_data(symbol)


def fetch_stock(symbol: str) -> pd.DataFrame | None:
    """Fetch weekly data for an NSE stock. Appends .NS if needed."""
    if not symbol.endswith(".NS"):
        symbol = symbol + ".NS"
    return fetch_weekly_data(symbol)


# ── Alignment helper ───────────────────────────────────────────────────────────

def align_data(stock_df: pd.DataFrame,
               bench_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align stock and benchmark DataFrames to same date index."""
    common_dates = stock_df.index.intersection(bench_df.index)
    return stock_df.loc[common_dates], bench_df.loc[common_dates]


# ── NSE Universe ───────────────────────────────────────────────────────────────

def get_nse_universe() -> list[str]:
    """Fetch live NSE stock universe via nsepython."""
    try:
        from nsepython import nse_eq_symbols
        print("  Fetching NSE stock universe...")
        symbols = nse_eq_symbols()
        print(f"  Got {len(symbols)} symbols from NSE")
        return symbols
    except Exception as e:
        print(f"  ❌ Could not fetch NSE universe: {e}")
        return []


# ── Quick verification ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing data_fetcher.py...")

    print("\n1. Fetching benchmark ^CNX200...")
    bench = fetch_benchmark()
    if bench is not None:
        print(f"   ✅ Got {len(bench)} weeks")
    else:
        print("   ❌ Failed")

    print("\n2. Fetching RELIANCE...")
    stock = fetch_stock("RELIANCE")
    if stock is not None:
        print(f"   ✅ Got {len(stock)} weeks, "
              f"latest close: ₹{stock['Close'].iloc[-1]:,.2f}")
    else:
        print("   ❌ Failed")
