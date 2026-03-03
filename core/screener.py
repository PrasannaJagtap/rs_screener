"""
screener.py
-----------
Main screening engine.
Fetches all NSE stocks via nsepython, runs all conditions,
saves progress as it goes (resume if interrupted).
"""

import pandas as pd
import numpy as np
import os, sys, json, time, logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_fetcher import fetch_stock, fetch_benchmark, align_data
from core.rs_calculator import build_rs_summary
from core.conditions   import score_stock, DEFAULT_CONDITIONS


# ── Paths ─────────────────────────────────────────────────────────

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(BASE_DIR, "results", "manual")
LOG_DIR      = os.path.join(BASE_DIR, "logs")
PROGRESS_DIR = os.path.join(BASE_DIR, "results", "manual")

os.makedirs(RESULTS_DIR,  exist_ok=True)
os.makedirs(LOG_DIR,      exist_ok=True)

# ── Logger ────────────────────────────────────────────────────────

# def setup_logger(run_id: str) -> logging.Logger:
#     log_file = os.path.join(LOG_DIR, f"screener_{run_id}.log")
#     logger   = logging.getLogger("screener")
#     logger.setLevel(logging.INFO)

#     if not logger.handlers:
#         fh = logging.FileHandler(log_file)
#         fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
#         logger.addHandler(fh)

#     return logger

def setup_logger(run_id: str) -> logging.Logger:
    log_file = os.path.join(LOG_DIR, f"screener_{run_id}.log")
    logger   = logging.getLogger("screener")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        fh.setStream(open(log_file, "a"))   # explicit handle
        logger.addHandler(fh)

    return logger


# ── NSE Universe ──────────────────────────────────────────────────

def get_nse_universe() -> list[str]:
    """
    Fetch all NSE equity symbols via nsepython.
    Returns a list of clean ticker symbols (without .NS suffix).
    Falls back to a small hardcoded list if nsepython fails.
    """
    print("  Fetching NSE stock universe...")
    try:
        from nsepython import nse_eq_symbols
        symbols = nse_eq_symbols()

        if isinstance(symbols, list) and len(symbols) > 100:
            print(f"  Got {len(symbols)} symbols from NSE")
            return symbols
        else:
            raise ValueError("Too few symbols returned")

    except Exception as e:
        print(f"  nsepython failed ({e}) — using Nifty 500 fallback")
        # Fallback: top Nifty 500 stocks (partial list for safety)
        fallback = [
            "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK",
            "HINDUNILVR","SBIN","BAJFINANCE","BHARTIARTL","KOTAKBANK",
            "LT","AXISBANK","ASIANPAINT","MARUTI","SUNPHARMA",
            "TITAN","ULTRACEMCO","NESTLEIND","WIPRO","TECHM",
            "HCLTECH","POWERGRID","NTPC","ONGC","JSWSTEEL",
            "TATASTEEL","ADANIENT","ADANIPORTS","COALINDIA","BAJAJ_AUTO",
        ]
        print(f"  Fallback list: {len(fallback)} stocks")
        return fallback


# ── Progress Save/Load ────────────────────────────────────────────

# def save_progress(run_id: str, completed: list, results: list):
#     path = os.path.join(PROGRESS_DIR, f"progress_{run_id}.json")
#     with open(path, "w") as f:
#         json.dump({"completed": completed, "results": results}, f)

def save_progress(run_id: str, completed: list, results: list):
    path = os.path.join(PROGRESS_DIR, f"progress_{run_id}.json")
    with open(path, "w") as f:
        json.dump({"completed": completed, "results": results}, f)
        f.flush()
        os.fsync(f.fileno())


def load_progress(run_id: str) -> tuple[list, list]:
    path = os.path.join(PROGRESS_DIR, f"progress_{run_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"  Resuming from saved progress: {len(data['completed'])} done")
        return data["completed"], data["results"]
    return [], []


def delete_progress(run_id: str):
    path = os.path.join(PROGRESS_DIR, f"progress_{run_id}.json")
    if os.path.exists(path):
        os.remove(path)


# ── Progress Bar ──────────────────────────────────────────────────

def print_progress(current: int, total: int, start_time: float,
                   found_t1: int, found_t2: int, symbol: str):
    elapsed   = time.time() - start_time
    pct       = current / total
    bar_len   = 30
    filled    = int(bar_len * pct)
    bar       = "█" * filled + "░" * (bar_len - filled)

    if current > 0:
        eta_secs = (elapsed / current) * (total - current)
        eta_str  = f"{int(eta_secs // 60)}m {int(eta_secs % 60)}s"
    else:
        eta_str = "--"

    print(
        f"\r  [{bar}] {current}/{total} ({pct*100:.1f}%)"
        f"  ETA: {eta_str}"
        f"  T1: {found_t1}  T2: {found_t2}"
        f"  Now: {symbol:<12}",
        end="", flush=True
    )


# ── Single Stock Processor ────────────────────────────────────────

def process_stock(
    symbol:    str,
    bench_df:  pd.DataFrame,
    conditions: dict,
    rs_period:  int,
    ma_period:  int,
    sma_period: int,
    logger:    logging.Logger,
) -> dict | None:
    """
    Fetch, calculate RS, score conditions for one stock.
    Returns a result dict or None if the stock should be skipped.
    """
    try:
        stock_df = fetch_stock(symbol)
        if stock_df is None:
            logger.warning(f"SKIP | {symbol} | no data returned")
            return None

        # Need enough data for RS calculation
        if len(stock_df) < rs_period + ma_period + 10:
            logger.warning(f"SKIP | {symbol} | insufficient data ({len(stock_df)} weeks)")
            return None

        s_aligned, b_aligned = align_data(stock_df, bench_df)

        if len(s_aligned) < rs_period + ma_period + 10:
            logger.warning(f"SKIP | {symbol} | insufficient aligned data")
            return None

        summary = build_rs_summary(
            stock_close  = s_aligned["Close"],
            bench_close  = b_aligned["Close"],
            rs_period    = rs_period,
            ma_period    = ma_period,
            sma_period   = sma_period,
            slope_period = 5,
        )

        report = score_stock(
            summary       = summary,
            stock_df      = stock_df,
            conditions    = conditions,
            min_threshold = 4,
        )

        latest = summary.iloc[-1]

        return {
            "symbol"       : symbol,
            "close"        : round(float(latest["Close"]), 2),
            "rs"           : round(float(latest["RS"]), 4),
            "rs_ma"        : round(float(latest["RS_MA"]), 4),
            "rs_slope"     : round(float(latest["RS_Slope"]), 4),
            "price_sma20"  : round(float(latest["Price_SMA_20"]), 2),
            "price_wma200" : round(float(latest["Price_WMA_200"]), 2) if not pd.isna(latest["Price_WMA_200"]) else None,
            "score"        : report["score"],
            "total_active" : report["total_active"],
            "tier"         : report["tier"],
            "conditions"   : {k: bool(v) for k, v in report["conditions"].items()},
            "run_date"     : datetime.today().strftime("%Y-%m-%d"),
        }

    except Exception as e:
        logger.error(f"ERROR | {symbol} | {type(e).__name__}: {e}")
        print(f"\n  ⚠️  {symbol}: {e}")
        return None


# ── Main Screener ─────────────────────────────────────────────────

def run_screener(
    benchmark:  str  = "^CNX200",
    conditions: dict = None,
    rs_period:  int  = 52,
    ma_period:  int  = 20,
    sma_period: int  = 20,
    run_id:     str  = None,
    delay_secs: float = 0.3,
) -> pd.DataFrame:
    """
    Run the full NSE screener.

    Parameters
    ----------
    benchmark   : yfinance ticker for benchmark index
    conditions  : dict of condition flags (DEFAULT_CONDITIONS if None)
    rs_period   : RS lookback in weeks
    ma_period   : RS MA period
    sma_period  : Price SMA period
    run_id      : Unique ID for this run (auto-generated if None)
    delay_secs  : Pause between downloads to avoid rate limiting

    Returns
    -------
    pd.DataFrame of all results sorted by RS descending
    """
    if conditions is None:
        conditions = DEFAULT_CONDITIONS

    if run_id is None:
        run_id = datetime.today().strftime("%Y%m%d_%H%M")

    # Raise macOS open file limit
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 4096), hard))

    logger = setup_logger(run_id)
    logger.info(f"Run started | benchmark={benchmark} | rs_period={rs_period}")

    print("\n" + "=" * 65)
    print(f"  RS SCREENER — RUN ID: {run_id}")
    print(f"  Benchmark : {benchmark}")
    print(f"  RS Period : {rs_period} weeks")
    print(f"  MA Period : {ma_period} weeks")
    print("=" * 65)

    # ── Step 1: Benchmark ─────────────────────────────────────────
    print("\n📊 Fetching benchmark...")
    bench_df = fetch_benchmark(benchmark)
    if bench_df is None:
        print("❌ Could not fetch benchmark. Aborting.")
        return pd.DataFrame()
    print(f"  ✅ {benchmark}: {len(bench_df)} weeks")

    # ── Step 2: Universe ──────────────────────────────────────────
    print("\n📋 Getting NSE universe...")
    universe = get_nse_universe()
    total    = len(universe)
    print(f"  ✅ {total} stocks to screen")

    # ── Step 3: Load progress (resume if interrupted) ─────────────
    completed, results = load_progress(run_id)
    completed_set      = set(completed)

    remaining = [s for s in universe if s not in completed_set]
    print(f"\n⚙️  Starting screening...")
    if completed:
        print(f"  Resuming: {len(completed)} already done, {len(remaining)} remaining")

    # ── Step 4: Main loop ─────────────────────────────────────────
    start_time = time.time()
    found_t1   = sum(1 for r in results if r["tier"] == "Tier 1")
    found_t2   = sum(1 for r in results if r["tier"] == "Tier 2")

    print()  # blank line before progress bar

    for i, symbol in enumerate(remaining):
        current = len(completed) + i + 1

        print_progress(current, total, start_time, found_t1, found_t2, symbol)

        result = process_stock(
            symbol     = symbol,
            bench_df   = bench_df,
            conditions = conditions,
            rs_period  = rs_period,
            ma_period  = ma_period,
            sma_period = sma_period,
            logger     = logger,
        )

        if result is not None:
            results.append(result)
            if result["tier"] == "Tier 1":
                found_t1 += 1
            elif result["tier"] == "Tier 2":
                found_t2 += 1
            logger.info(f"OK | {symbol} | RS={result['rs']:+.4f} | {result['tier']} | score={result['score']}/{result['total_active']}")

        completed.append(symbol)

        # Save progress every 50 stocks
        if len(completed) % 50 == 0:
            save_progress(run_id, completed, results)

        time.sleep(delay_secs)

    print()  # newline after progress bar

    # ── Step 5: Build results DataFrame ───────────────────────────
    df = pd.DataFrame(results)

    if df.empty:
        print("\n⚠️  No results. Check logs.")
        return df

    df = df.sort_values("rs", ascending=False).reset_index(drop=True)

    # ── Step 6: Save final CSV ────────────────────────────────────
    out_file = os.path.join(RESULTS_DIR, f"screener_{run_id}.csv")
    df.to_csv(out_file, index=False)

    # ── Step 7: Clean up progress file ───────────────────────────
    delete_progress(run_id)

    # ── Step 8: Print summary ─────────────────────────────────────
    elapsed = time.time() - start_time
    tier1   = df[df["tier"] == "Tier 1"]
    tier2   = df[df["tier"] == "Tier 2"]
    watch   = df[df["tier"] == "Watchlist"]

    print("\n" + "=" * 65)
    print(f"  SCREENING COMPLETE  ({int(elapsed//60)}m {int(elapsed%60)}s)")
    print("=" * 65)
    print(f"  Total screened : {len(completed)}")
    print(f"  Results found  : {len(df)}")
    print(f"  Tier 1         : {len(tier1)}  (passes ALL conditions)")
    print(f"  Tier 2         : {len(tier2)}  (passes >= 4 conditions)")
    print(f"  Watchlist      : {len(watch)}  (one condition away)")
    print(f"  Output saved   : {out_file}")
    print(f"  Log file       : {LOG_DIR}/screener_{run_id}.log")

    # ── Step 9: Print Tier 1 stocks ───────────────────────────────
    if not tier1.empty:
        print(f"\n  🏆 TIER 1 STOCKS ({len(tier1)} found):")
        print(f"  {'Symbol':<14} {'Close':>10}  {'RS':>8}  {'Score':>6}")
        print(f"  {'-'*50}")
        for _, row in tier1.iterrows():
            print(f"  {row['symbol']:<14} ₹{row['close']:>9,.2f}  {row['rs']:>+8.4f}  {row['score']}/{row['total_active']}")
    else:
        print("\n  No Tier 1 stocks found with current conditions.")

    if not tier2.empty:
        print(f"\n  📈 TOP 10 TIER 2 STOCKS:")
        print(f"  {'Symbol':<14} {'Close':>10}  {'RS':>8}  {'Score':>6}")
        print(f"  {'-'*50}")
        for _, row in tier2.head(10).iterrows():
            print(f"  {row['symbol']:<14} ₹{row['close']:>9,.2f}  {row['rs']:>+8.4f}  {row['score']}/{row['total_active']}")

    logger.info(f"Run complete | T1={len(tier1)} | T2={len(tier2)} | Watch={len(watch)}")
    return df


# ── Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":

    # Quick test — runs only first 20 stocks to verify pipeline
    # Change TEST_MODE = False to run full 2000-stock universe

    TEST_MODE = False

    if TEST_MODE:
        print("\n" + "="*65)
        print("  TEST MODE — Running first 20 stocks only")
        print("  Set TEST_MODE = False for full run")
        print("="*65)

        from nsepython import nse_eq_symbols
        test_symbols = nse_eq_symbols()[:20]

        bench_df = fetch_benchmark("^CNX200")

        logger  = setup_logger("test")
        results = []

        for symbol in test_symbols:
            print(f"  Processing {symbol}...")
            result = process_stock(
                symbol     = symbol,
                bench_df   = bench_df,
                conditions = DEFAULT_CONDITIONS,
                rs_period  = 52,
                ma_period  = 20,
                sma_period = 20,
                logger     = logger,
            )
            if result:
                results.append(result)
            time.sleep(0.3)

        if results:
            df = pd.DataFrame(results).sort_values("rs", ascending=False)
            print("\n" + "="*65)
            print("  TEST RESULTS")
            print("="*65)
            print(f"  {'Symbol':<14} {'Close':>10}  {'RS':>8}  {'Score':>6}  Tier")
            print(f"  {'-'*60}")
            for _, row in df.iterrows():
                print(f"  {row['symbol']:<14} ₹{row['close']:>9,.2f}  {row['rs']:>+8.4f}  {row['score']}/{row['total_active']}  {row['tier']}")

            print(f"\n  Tier 1: {len(df[df['tier']=='Tier 1'])}")
            print(f"  Tier 2: {len(df[df['tier']=='Tier 2'])}")
            print(f"  Watchlist: {len(df[df['tier']=='Watchlist'])}")
            print("\n  Test passed. Run with TEST_MODE=False for full universe.")
    else:
        # Full run
        results_df = run_screener(
            benchmark  = "^CNX200",
            conditions = DEFAULT_CONDITIONS,
            rs_period  = 52,
            ma_period  = 20,
            sma_period = 20,
            delay_secs = 0.3,
        )