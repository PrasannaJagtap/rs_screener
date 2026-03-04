"""
crossover_scanner.py
--------------------
Finds NSE stocks where RS has recently crossed above RS MA (from below).
This is an early signal — stock starting to outperform the benchmark.

Output columns:
  symbol        : NSE ticker
  cross_weeks_ago: how many weeks ago the crossover happened (1, 2 or 3)
  rs_now        : current RS value
  rs_ma_now     : current RS MA value
  rs_gap        : rs_now - rs_ma_now (how far above MA it is now)
  close         : latest closing price
  sector        : broad sector tag
  sustained     : True if RS has stayed above MA since crossover

Usage:
  python3 core/crossover_scanner.py              # scan last 3 weeks
  python3 core/crossover_scanner.py --weeks 2    # scan last 2 weeks
  python3 core/crossover_scanner.py --weeks 1    # only this week's crossover
"""

import pandas as pd
import numpy as np
import os, sys, argparse, time
from datetime import datetime

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "manual")

# Ensure BASE_DIR is on path for both direct execution and when imported as module
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    # When imported as core.crossover_scanner (from app.py)
    from core.data_fetcher  import fetch_stock, fetch_benchmark, align_data, get_nse_universe
    from core.rs_calculator import build_rs_summary
    from core.scorer        import assign_sector
except ImportError:
    # When run directly: python3 core/crossover_scanner.py
    from data_fetcher  import fetch_stock, fetch_benchmark, align_data, get_nse_universe
    from rs_calculator import build_rs_summary
    from scorer        import assign_sector


# ── Core crossover detection ───────────────────────────────────────

def find_crossover(summary: pd.DataFrame, lookback: int = 3) -> dict | None:
    """
    Check if RS crossed above RS MA within the last `lookback` weeks.

    Returns a dict with crossover details, or None if no crossover found.

    A valid crossover is:
      - Week N-1 : RS was BELOW RS MA  (rs <= rs_ma)
      - Week N   : RS is ABOVE RS MA   (rs > rs_ma)
    """
    rs    = summary["RS"].values
    rs_ma = summary["RS_MA"].values

    if len(rs) < lookback + 2:
        return None

    for weeks_ago in range(1, lookback + 1):
        idx_now  = -(weeks_ago)        # the crossover week
        idx_prev = -(weeks_ago + 1)    # week before crossover

        rs_now   = rs[idx_now]
        rs_prev  = rs[idx_prev]
        ma_now   = rs_ma[idx_now]
        ma_prev  = rs_ma[idx_prev]

        # Crossover: was below, now above
        if rs_prev <= ma_prev and rs_now > ma_now:

            # Check if sustained (RS stayed above MA from crossover to today)
            sustained = True
            for j in range(1, weeks_ago):
                if rs[-j] <= rs_ma[-j]:
                    sustained = False
                    break

            return {
                "cross_weeks_ago" : weeks_ago,
                "rs_at_cross"     : round(float(rs_now), 4),
                "rs_now"          : round(float(rs[-1]), 4),
                "rs_ma_now"       : round(float(rs_ma[-1]), 4),
                "rs_gap"          : round(float(rs[-1] - rs_ma[-1]), 4),
                "sustained"       : sustained,
                "rs_slope"        : round(float(rs[-1] - rs[-4]), 4)
                                    if len(rs) >= 4 else 0.0,
            }

    return None


# ── Main scanner ───────────────────────────────────────────────────

def run_crossover_scanner(
    lookback   : int   = 3,
    benchmark  : str   = "^CNX200",
    rs_period  : int   = 52,
    ma_period  : int   = 20,
    delay_secs : float = 0.3,
    min_rs     : float = -0.5,      # skip deeply negative RS stocks
) -> pd.DataFrame:
    """
    Scan entire NSE universe for recent RS/RSMA crossovers.

    Parameters
    ----------
    lookback   : how many weeks back to look for crossovers (1-5)
    min_rs     : minimum RS value at crossover (filter out junk)

    Returns
    -------
    DataFrame of stocks with crossover, sorted by rs_gap descending
    """

    print("=" * 60)
    print(f"  RS CROSSOVER SCANNER")
    print(f"  Looking back: {lookback} weeks")
    print(f"  Benchmark   : {benchmark}")
    print(f"  RS Period   : {rs_period}w  |  MA Period: {ma_period}w")
    print("=" * 60)

    # Fetch benchmark
    print("\n📊 Fetching benchmark...")
    bench_raw = fetch_benchmark(benchmark)
    if bench_raw is None:
        print("  ❌ Could not fetch benchmark")
        return pd.DataFrame()
    print(f"  ✅ {benchmark}: {len(bench_raw)} weeks")

    # Get universe
    print("\n📋 Getting NSE universe...")
    symbols = get_nse_universe()
    print(f"  ✅ {len(symbols)} stocks to scan\n")

    results  = []
    skipped  = 0
    total    = len(symbols)

    for i, symbol in enumerate(symbols):
        ticker = f"{symbol}.NS"

        # Progress bar
        pct  = (i + 1) / total * 100
        done = int(pct / 3)
        bar  = "█" * done + "░" * (20 - done)
        found_so_far = len(results)
        print(
            f"\r  [{bar}] {i+1}/{total} ({pct:.1f}%)  "
            f"Found: {found_so_far}  Now: {symbol:<12}",
            end="", flush=True
        )

        # Fetch stock
        stock_raw = fetch_stock(symbol)
        if stock_raw is None or len(stock_raw) < 65:
            skipped += 1
            time.sleep(delay_secs)
            continue

        # Align and build RS summary
        try:
            s_aligned, b_aligned = align_data(stock_raw, bench_raw)
            summary = build_rs_summary(
                stock_close  = s_aligned["Close"],
                bench_close  = b_aligned["Close"],
                rs_period    = rs_period,
                ma_period    = ma_period,
                sma_period   = 20,
                slope_period = 5,
            )
        except Exception:
            skipped += 1
            time.sleep(delay_secs)
            continue

        if summary is None or len(summary) < ma_period + 5:
            skipped += 1
            time.sleep(delay_secs)
            continue

        # Check for crossover
        cross = find_crossover(summary, lookback=lookback)

        if cross and cross["rs_at_cross"] >= min_rs:
            latest = summary.iloc[-1]
            results.append({
                "symbol"          : symbol,
                "cross_weeks_ago" : cross["cross_weeks_ago"],
                "rs_now"          : cross["rs_now"],
                "rs_ma_now"       : cross["rs_ma_now"],
                "rs_gap"          : cross["rs_gap"],
                "rs_at_cross"     : cross["rs_at_cross"],
                "rs_slope"        : cross["rs_slope"],
                "sustained"       : cross["sustained"],
                "close"           : round(float(latest["Close"]), 2),
                "sector"          : assign_sector(symbol),
            })

        time.sleep(delay_secs)

    print(f"\n\n  Scan complete — {len(results)} crossovers found "
          f"({skipped} skipped)\n")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Sort: sustained crossovers first, then by rs_gap (strength above MA)
    df = df.sort_values(
        ["sustained", "rs_gap"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return df


# ── Print results ──────────────────────────────────────────────────

def print_results(df: pd.DataFrame, lookback: int):

    if df.empty:
        print("  No crossovers found.")
        return

    sustained = df[df["sustained"] == True]
    fresh     = df[df["sustained"] == False]

    print("=" * 70)
    print(f"  RS CROSSOVER RESULTS — Last {lookback} weeks")
    print(f"  Total found : {len(df)}")
    print(f"  Sustained   : {len(sustained)}  (RS stayed above MA)")
    print(f"  Fresh only  : {len(fresh)}  (crossover but not yet sustained)")
    print("=" * 70)

    # Group by weeks ago
    for weeks in range(1, lookback + 1):
        group = df[df["cross_weeks_ago"] == weeks]
        if group.empty:
            continue

        label = "THIS WEEK" if weeks == 1 else f"{weeks} WEEKS AGO"
        print(f"\n  📍 {label} ({len(group)} stocks)")
        print(f"  {'Symbol':<14} {'RS Now':>8} {'RS MA':>8} "
              f"{'Gap':>8} {'Slope':>8} {'Close':>10}  "
              f"{'Sustained':<10} Sector")
        print(f"  {'-'*80}")

        for _, row in group.iterrows():
            sustained_tag = "✅ Yes" if row["sustained"] else "⏳ New"
            print(
                f"  {row['symbol']:<14}"
                f" {row['rs_now']:>+8.4f}"
                f" {row['rs_ma_now']:>8.4f}"
                f" {row['rs_gap']:>+8.4f}"
                f" {row['rs_slope']:>+8.4f}"
                f" ₹{row['close']:>9,.2f}"
                f"  {sustained_tag:<10}"
                f" {row['sector']}"
            )

    # Sector breakdown
    print(f"\n  SECTOR BREAKDOWN")
    print(f"  {'-'*40}")
    sector_counts = df["sector"].value_counts()
    for sector, count in sector_counts.items():
        bar = "█" * count
        print(f"  {sector:<18} {count:>3}  {bar}")


# ── Top picks filter ───────────────────────────────────────────────

def print_top_picks(df: pd.DataFrame, top_n: int = 20):
    """
    Apply strict quality filters and print the best actionable setups.

    Filter rules:
      1. Sustained = True   (RS held above MA since crossover — no failed crosses)
      2. RS Now > 0         (actually outperforming the benchmark right now)
      3. RS Gap > 0.02      (meaningful distance above MA, not just barely above)
      4. RS Slope > 0       (RS is still rising, not rolling over)

    Scored and ranked by a composite score:
      score = rs_now * 0.4 + rs_gap * 0.4 + rs_slope * 0.2
    """

    if df.empty:
        return

    picks = df[
        (df["sustained"] == True)  &
        (df["rs_now"]    >  0.0)   &
        (df["rs_gap"]    >  0.02)  &
        (df["rs_slope"]  >  0.0)
    ].copy()

    if picks.empty:
        print("\n  ⚠️  No stocks passed all quality filters this week.")
        print("  Try relaxing --min-rs or --weeks to cast a wider net.")
        return

    # Composite score
    picks["score"] = (
        picks["rs_now"]   * 0.4 +
        picks["rs_gap"]   * 0.4 +
        picks["rs_slope"] * 0.2
    )
    picks = picks.sort_values("score", ascending=False).head(top_n)

    # Conviction tags
    def conviction(row):
        if row["rs_now"] > 0.5 and row["rs_gap"] > 0.1 and row["rs_slope"] > 0.1:
            return "🔥 HIGH"
        elif row["rs_now"] > 0.2 and row["rs_gap"] > 0.05:
            return "⭐ MED"
        else:
            return "👀 WATCH"

    picks["conviction"] = picks.apply(conviction, axis=1)

    print("\n")
    print("█" * 70)
    print(f"  🏆 TOP PICKS — {len(picks)} ACTIONABLE SETUPS")
    print(f"  Filter: Sustained ✅ | RS > 0 | Gap > 0.02 | Slope rising")
    print(f"  Ranked by: RS strength + gap above MA + momentum slope")
    print("█" * 70)

    print(f"\n  {'#':<4} {'Symbol':<14} {'Conviction':<10} "
          f"{'RS Now':>8} {'Gap':>8} {'Slope':>8} "
          f"{'Crossed':>10} {'Close':>10}  Sector")
    print(f"  {'-'*85}")

    for rank, (_, row) in enumerate(picks.iterrows(), 1):
        weeks_label = (
            "This week" if row["cross_weeks_ago"] == 1
            else f"{int(row['cross_weeks_ago'])}w ago"
        )
        print(
            f"  {rank:<4}"
            f" {row['symbol']:<14}"
            f" {row['conviction']:<10}"
            f" {row['rs_now']:>+8.4f}"
            f" {row['rs_gap']:>+8.4f}"
            f" {row['rs_slope']:>+8.4f}"
            f" {weeks_label:>10}"
            f" ₹{row['close']:>9,.2f}"
            f"  {row['sector']}"
        )

    # Conviction summary
    high  = len(picks[picks["conviction"] == "🔥 HIGH"])
    med   = len(picks[picks["conviction"] == "⭐ MED"])
    watch = len(picks[picks["conviction"] == "👀 WATCH"])

    print(f"\n  Conviction breakdown:")
    print(f"    🔥 HIGH  : {high}  (strong RS, big gap, rising slope — act first)")
    print(f"    ⭐ MED   : {med}  (solid setup, monitor closely)")
    print(f"    👀 WATCH : {watch}  (early stage, wait for confirmation)")

    # Sector spread of top picks
    print(f"\n  Sectors in top picks:")
    for sector, count in picks["sector"].value_counts().items():
        print(f"    {sector:<18} {count} stock{'s' if count > 1 else ''}")

    print(f"\n  💡 Next step: Check each stock in Stock Detail tab")
    print(f"     Look for: RS chart turning up + price above SMA20 + volume")
    print("█" * 70)

    return picks


# ── Save results ───────────────────────────────────────────────────

def save_results(df: pd.DataFrame, lookback: int) -> str:
    ts       = datetime.today().strftime("%Y%m%d_%H%M")
    filename = f"crossover_{lookback}w_{ts}.csv"
    path     = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False)
    return path


def save_top_picks(picks: pd.DataFrame, lookback: int) -> str:
    ts       = datetime.today().strftime("%Y%m%d_%H%M")
    filename = f"top_picks_{lookback}w_{ts}.csv"
    path     = os.path.join(RESULTS_DIR, filename)
    picks.to_csv(path, index=False)
    return path


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RS Crossover Scanner")
    parser.add_argument(
        "--weeks", type=int, default=3,
        help="How many weeks back to look for crossovers (default: 3)"
    )
    parser.add_argument(
        "--min-rs", type=float, default=-0.5,
        help="Minimum RS value at crossover (default: -0.5)"
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Number of top picks to show (default: 20)"
    )
    args = parser.parse_args()

    lookback = max(1, min(args.weeks, 8))

    df = run_crossover_scanner(
        lookback   = lookback,
        min_rs     = args.min_rs,
    )

    if not df.empty:
        # Full results (all crossovers)
        print_results(df, lookback)

        # Save full results CSV
        path = save_results(df, lookback)
        print(f"\n  📄 Full results saved: {os.path.basename(path)}")

        # ── TOP PICKS ADD-ON ───────────────────────────────────────
        picks = print_top_picks(df, top_n=args.top)

        # Save top picks CSV separately
        if picks is not None and not picks.empty:
            picks_path = save_top_picks(picks, lookback)
            print(f"  📄 Top picks saved : {os.path.basename(picks_path)}")

        print(f"\n  💡 TIP: Load either CSV in the Stock Detail tab")
        print(f"         to see the RS chart for each stock.")
