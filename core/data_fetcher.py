"""
data_fetcher.py
---------------
Fetches weekly OHLCV price data for:
  - NSE benchmark index (via nsepython — no rate limits)
  - NSE stocks (via yfinance)

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

# nsepython index symbol mapping
NSE_INDEX_MAP = {
    "^CNX200"   : "NIFTY 200",
    "^NSEI"     : "NIFTY 50",
    "^NSEBANK"  : "NIFTY BANK",
    "^NSEMDCP50": "NIFTY MIDCAP 50",
}


# ── Benchmark fetch via nsepython ──────────────────────────────────────────────

def fetch_benchmark(symbol: str = DEFAULT_BENCHMARK) -> pd.DataFrame | None:
    """
    Fetch weekly benchmark index data directly from NSE via nsepython.
    No Yahoo Finance — no rate limits.

    Falls back to yfinance if nsepython fails.
    """
    nse_name = NSE_INDEX_MAP.get(symbol)

    if nse_name:
        try:
            df = _fetch_benchmark_nsepython(nse_name)
            if df is not None and len(df) >= 60:
                print(f"  ✅ {symbol}: {len(df)} weeks (via NSE)")
                return df
            else:
                print(f"  ⚠️  nsepython returned short data, falling back to yfinance")
        except Exception as e:
            print(f"  ⚠️  nsepython failed ({e}), falling back to yfinance")

    # Fallback to yfinance
    print(f"  Trying yfinance for {symbol}...")
    return fetch_weekly_data(symbol)


def _fetch_benchmark_nsepython(index_name: str,
                                weeks: int = WEEKS_REQUIRED) -> pd.DataFrame | None:
    """
    Fetch daily index history from NSE via nsepython and resample to weekly.
    """
    from nsepython import index_history

    start_date = datetime.today() - timedelta(days=weeks * 7 + 30)
    end_date   = datetime.today()

    start_str  = start_date.strftime("%d-%m-%Y")
    end_str    = end_date.strftime("%d-%m-%Y")

    raw = index_history(index_name, start_str, end_str)

    if raw is None or raw.empty:
        return None

    # nsepython returns columns like: HistoricalDate, OPEN, HIGH, LOW, CLOSE, ...
    raw.columns = [c.strip().upper() for c in raw.columns]

    # Find date column
    date_col = None
    for c in raw.columns:
        if "DATE" in c or "TIME" in c:
            date_col = c
            break
    if date_col is None:
        return None

    raw[date_col] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=[date_col])
    raw = raw.set_index(date_col).sort_index()

    # Rename to standard OHLCV
    col_map = {}
    for c in raw.columns:
        if c in ("OPEN", "OPEN INDEX VALUE"):
            col_map[c] = "Open"
        elif c in ("HIGH", "HIGH INDEX VALUE"):
            col_map[c] = "High"
        elif c in ("LOW", "LOW INDEX VALUE"):
            col_map[c] = "Low"
        elif c in ("CLOSE", "CLOSING INDEX VALUE", "CLOSE INDEX VALUE"):
            col_map[c] = "Close"
    raw = raw.rename(columns=col_map)

    needed = [c for c in ["Open", "High", "Low", "Close"] if c in raw.columns]
    if "Close" not in needed:
        return None

    raw = raw[needed].copy()

    # Convert to numeric
    for col in raw.columns:
        raw[col] = pd.to_numeric(
            raw[col].astype(str).str.replace(",", ""), errors="coerce"
        )
    raw = raw.dropna(subset=["Close"])

    # Resample daily → weekly (week ending Friday)
    weekly = raw.resample("W-FRI").agg({
        "Open"  : "first",
        "High"  : "max",
        "Low"   : "min",
        "Close" : "last",
    }).dropna(subset=["Close"])

    return weekly


# ── Core yfinance fetch (for stocks) ──────────────────────────────────────────

def fetch_weekly_data(ticker: str, weeks: int = WEEKS_REQUIRED) -> pd.DataFrame | None:
    """
    Fetch weekly OHLCV data for a given ticker via yfinance.
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


# ── Stock fetch ────────────────────────────────────────────────────────────────

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
    print("Testing data_fetcher.py...\n")

    print("1. Fetching benchmark ^CNX200 via nsepython...")
    bench = fetch_benchmark()
    if bench is not None:
        print(f"   ✅ Got {len(bench)} weeks, "
              f"latest close: {bench['Close'].iloc[-1]:,.2f}\n")
    else:
        print("   ❌ Failed\n")

    print("2. Fetching RELIANCE via yfinance...")
    stock = fetch_stock("RELIANCE")
    if stock is not None:
        print(f"   ✅ Got {len(stock)} weeks, "
              f"latest close: ₹{stock['Close'].iloc[-1]:,.2f}")
    else:
        print("   ❌ Failed")
