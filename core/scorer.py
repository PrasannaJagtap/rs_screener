"""
scorer.py
---------
Loads raw screener CSV, cleans data issues, and adds:
  - Data quality flags  : RS too high, penny stocks, price spikes
  - RS Rank             : percentile rank 0-100 (like IBD RS Rating)
  - Momentum Grade      : A / B / C / D
  - Sector tags         : broad sector from symbol mapping
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "manual")


# ── Data Quality Flags ─────────────────────────────────────────────

def flag_data_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows with likely data quality issues.
    Does NOT remove them — adds 'data_flag' and 'is_clean' columns.
    User can filter these out in the UI.
    """
    df = df.copy()
    flags = []

    for _, row in df.iterrows():
        f = []
        if row["rs"] > 5.0:
            f.append("RS_TOO_HIGH")      # unadjusted split/bonus
        if row["rs"] < -0.95:
            f.append("RS_TOO_LOW")       # near-suspended stock
        if row["close"] < 20:
            f.append("PENNY_STOCK")      # below Rs.20
        flags.append("|".join(f) if f else "")

    df["data_flag"] = flags
    df["is_clean"]  = df["data_flag"] == ""
    return df


# ── RS Percentile Rank ─────────────────────────────────────────────

def add_rs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    RS Rank = percentile rank among all clean stocks (0-100).
    RS Rank 95 means the stock beats 95% of the universe in RS.
    Similar to IBD's RS Rating.
    """
    df         = df.copy()
    df["rs_rank"] = np.nan
    clean_mask = df["is_clean"]

    if clean_mask.sum() > 0:
        ranks = df.loc[clean_mask, "rs"].rank(pct=True) * 100
        df.loc[clean_mask, "rs_rank"] = ranks.round(1)

    return df


# ── Momentum Grade ─────────────────────────────────────────────────

def add_momentum_grade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Grade based on RS Rank + Tier:
      A = RS Rank >= 80 AND Tier 1        → Top leaders
      B = RS Rank >= 60 OR Tier 1         → Strong stocks
      C = RS Rank >= 40 OR Tier 2         → Watch candidates
      D = Everything else                 → Avoid
    """
    df = df.copy()

    def grade(row):
        if not row["is_clean"]:
            return "?"
        rs_rank = row.get("rs_rank") or 0
        tier    = row.get("tier", "")
        score   = row.get("score", 0)
        total   = row.get("total_active", 7)
        pct     = score / total if total > 0 else 0

        if rs_rank >= 80 and tier == "Tier 1":
            return "A"
        elif rs_rank >= 60 or (tier == "Tier 1" and pct >= 0.85):
            return "B"
        elif rs_rank >= 40 or tier == "Tier 2":
            return "C"
        else:
            return "D"

    df["grade"] = df.apply(grade, axis=1)
    return df


# ── Sector Tags ────────────────────────────────────────────────────

SECTOR_OVERRIDES = {
    "SBIN": "Banking", "HDFCBANK": "Banking", "ICICIBANK": "Banking",
    "KOTAKBANK": "Banking", "AXISBANK": "Banking", "FEDERALBNK": "Banking",
    "RBLBANK": "Banking", "CANBK": "Banking", "INDIANB": "Banking",
    "BANKINDIA": "Banking", "UNIONBANK": "Banking", "MAHABANK": "Banking",
    "IDBI": "Banking", "PNB": "Banking", "BANDHANBNK": "Banking",
    "KARURVYSYA": "Banking", "KTKBANK": "Banking", "CUB": "Banking",
    "DCBBANK": "Banking", "IDFCFIRSTB": "Banking",
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "TECHM": "IT", "MPHASIS": "IT", "LTIM": "IT", "COFORGE": "IT",
    "HINDCOPPER": "Metals", "NATIONALUM": "Metals", "VEDL": "Metals",
    "SAIL": "Metals", "NMDC": "Metals", "HEG": "Metals",
    "GRAPHITE": "Metals", "JSWSTEEL": "Metals", "TATASTEEL": "Metals",
    "APLAPOLLO": "Metals", "JINDALSAW": "Metals", "JSL": "Metals",
    "GPIL": "Metals",
    "BEL": "Defence", "HAL": "Defence", "DATAPATTNS": "Defence",
    "POLYCAB": "Cables", "KEI": "Cables",
    "TITAN": "Consumer", "NYKAA": "Consumer",
    "BRITANNIA": "FMCG", "GODREJCP": "FMCG", "HINDUNILVR": "FMCG",
    "DABUR": "FMCG", "NESTLEIND": "FMCG",
    "HEROMOTOCO": "Auto", "TVSMOTOR": "Auto", "BAJAJ-AUTO": "Auto",
    "MARUTI": "Auto", "TATAMOTORS": "Auto", "MOTHERSON": "Auto",
    "MCX": "Exchange", "BSE": "Exchange",
    "ADANIPORTS": "Ports",
    "LT": "Infrastructure", "GMRAIRPORT": "Infrastructure",
    "SIEMENS": "Capital Goods", "ABB": "Capital Goods",
    "CUMMINSIND": "Capital Goods", "SCHNEIDER": "Capital Goods",
    "KIRLOSENG": "Capital Goods", "BHARATFORG": "Capital Goods",
    "TORNTPHARM": "Pharma", "GLENMARK": "Pharma", "LUPIN": "Pharma",
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "NTPC": "Power", "POWERGRID": "Power", "TORNTPOWER": "Power",
    "TATAPOWER": "Power", "ADANIGREEN": "Power",
    "ONGC": "Oil & Gas", "IOC": "Oil & Gas", "BPCL": "Oil & Gas",
    "OIL": "Oil & Gas", "MRPL": "Oil & Gas", "PETRONET": "Oil & Gas",
    "RELIANCE": "Oil & Gas",
    "SUNTV": "Media", "ZEEL": "Media",
    "BAJFINANCE": "Finance", "BAJAJFINSV": "Finance",
    "CHOLAFIN": "Finance", "MUTHOOTFIN": "Finance",
    "SBIN": "Banking",
}

SECTOR_KEYWORDS = {
    "BANK": "Banking", "FIN": "Finance", "PHARMA": "Pharma",
    "CHEM": "Chemicals", "STEEL": "Metals", "COPPER": "Metals",
    "ALUM": "Metals", "AUTO": "Auto", "MOTOR": "Auto", "TYRE": "Auto",
    "TECH": "IT", "SOFT": "IT", "INFO": "IT",
    "POWER": "Power", "ENERGY": "Power",
    "OIL": "Oil & Gas", "PETRO": "Oil & Gas",
    "DRUG": "Pharma", "MED": "Pharma",
    "CEMENT": "Cement", "INFRA": "Infrastructure",
    "DEFENCE": "Defence", "AGRO": "Agri", "CROP": "Agri",
}


def assign_sector(symbol: str) -> str:
    sym = symbol.upper()
    if sym in SECTOR_OVERRIDES:
        return SECTOR_OVERRIDES[sym]
    for keyword, sector in SECTOR_KEYWORDS.items():
        if keyword in sym:
            return sector
    return "Other"


def add_sectors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sector"] = df["symbol"].apply(assign_sector)
    return df


# ── Master Clean + Score ───────────────────────────────────────────

def clean_and_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all cleaning and scoring steps.
    Returns clean, ranked, graded DataFrame sorted by Grade then RS Rank.
    """
    df = flag_data_issues(df)
    df = add_rs_rank(df)
    df = add_momentum_grade(df)
    df = add_sectors(df)

    grade_order       = {"A": 0, "B": 1, "C": 2, "D": 3, "?": 4}
    df["grade_order"] = df["grade"].map(grade_order)
    df = df.sort_values(
        ["grade_order", "rs_rank"],
        ascending=[True, False]
    ).drop(columns=["grade_order"]).reset_index(drop=True)

    return df


# ── Load Latest CSV ────────────────────────────────────────────────

def load_latest_results() -> pd.DataFrame | None:
    files = [
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith("screener_") and f.endswith(".csv")
    ]
    if not files:
        return None
    files.sort(reverse=True)
    latest = os.path.join(RESULTS_DIR, files[0])
    print(f"  Loading: {files[0]}")
    return pd.read_csv(latest)


# ── Verification Test ──────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 65)
    print("  SCORER — VERIFICATION TEST")
    print("=" * 65)

    raw = load_latest_results()
    if raw is None:
        print("  No screener CSV found. Run screener.py first.")
        exit()

    print(f"  Raw results : {len(raw)} stocks")
    cleaned = clean_and_score(raw)

    # Summary stats
    clean   = cleaned[cleaned["is_clean"]]
    flagged = cleaned[~cleaned["is_clean"]]
    grade_a = cleaned[cleaned["grade"] == "A"]
    grade_b = cleaned[cleaned["grade"] == "B"]
    grade_c = cleaned[cleaned["grade"] == "C"]
    grade_d = cleaned[cleaned["grade"] == "D"]

    print(f"  Clean       : {len(clean)}")
    print(f"  Flagged     : {len(flagged)}")

    if len(flagged) > 0:
        flag_counts = flagged["data_flag"].value_counts().to_dict()
        for flag, count in flag_counts.items():
            print(f"    {flag:<20}: {count} stocks")

    print(f"\n  Grade A     : {len(grade_a)}  (Top leaders)")
    print(f"  Grade B     : {len(grade_b)}  (Strong)")
    print(f"  Grade C     : {len(grade_c)}  (Watch)")
    print(f"  Grade D     : {len(grade_d)}  (Avoid)")

    # Print Grade A stocks
    print(f"\n{'='*65}")
    print(f"  GRADE A STOCKS — Top Leaders")
    print(f"{'='*65}")
    print(f"  {'#':<4} {'Symbol':<14} {'RS Rank':>8} {'RS':>8} {'Close':>10}  {'Sector':<15} Tier")
    print(f"  {'-'*65}")

    for i, (_, row) in enumerate(grade_a.iterrows(), 1):
        print(
            f"  {i:<4} {row['symbol']:<14}"
            f" {row['rs_rank']:>8.1f}"
            f" {row['rs']:>+8.4f}"
            f" ₹{row['close']:>9,.2f}"
            f"  {row['sector']:<15}"
            f" {row['tier']}"
        )

    # Print Grade B top 15
    print(f"\n{'='*65}")
    print(f"  GRADE B STOCKS — Top 15")
    print(f"{'='*65}")
    print(f"  {'#':<4} {'Symbol':<14} {'RS Rank':>8} {'RS':>8} {'Close':>10}  {'Sector':<15} Tier")
    print(f"  {'-'*65}")

    for i, (_, row) in enumerate(grade_b.head(15).iterrows(), 1):
        print(
            f"  {i:<4} {row['symbol']:<14}"
            f" {row['rs_rank']:>8.1f}"
            f" {row['rs']:>+8.4f}"
            f" ₹{row['close']:>9,.2f}"
            f"  {row['sector']:<15}"
            f" {row['tier']}"
        )

    # Sector breakdown of Grade A+B
    print(f"\n{'='*65}")
    print(f"  SECTOR BREAKDOWN (Grade A + B)")
    print(f"{'='*65}")
    top_stocks  = cleaned[cleaned["grade"].isin(["A", "B"])]
    sector_dist = top_stocks["sector"].value_counts()
    for sector, count in sector_dist.items():
        bar = "█" * count
        print(f"  {sector:<20} {count:>3}  {bar}")

    # Save scored CSV
    out_file = os.path.join(
        RESULTS_DIR,
        f"scored_{datetime.today().strftime('%Y%m%d_%H%M')}.csv"
    )
    cleaned.to_csv(out_file, index=False)
    print(f"\n  Saved: {out_file}")
    print(f"  Next step: app.py — Streamlit UI")
