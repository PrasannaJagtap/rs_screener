"""
telegram.py
-----------
Sends RS Screener results to Telegram.

Message includes:
  - Top 5 RS leaders with price
  - Grade A stocks list
  - Sector breakdown summary
"""

import os, sys, glob
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "manual")
sys.path.insert(0, BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")


# ── Send message ───────────────────────────────────────────────────

def send_message(text: str) -> bool:
    """Send a text message via Telegram Bot API."""
    try:
        import requests
        url  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id"    : CHAT_ID,
            "text"       : text,
            "parse_mode" : "HTML",
        }, timeout=15)
        return resp.status_code == 200
    except Exception as e:
        print(f"  Telegram error: {e}")
        return False


# ── Load latest scored CSV ─────────────────────────────────────────

def load_latest_scored() -> pd.DataFrame | None:
    files = sorted(
        glob.glob(os.path.join(RESULTS_DIR, "scored_*.csv")),
        reverse=True
    )
    if not files:
        files = sorted(
            glob.glob(os.path.join(RESULTS_DIR, "screener_*.csv")),
            reverse=True
        )
    if not files:
        return None
    return pd.read_csv(files[0])


# ── Build message ──────────────────────────────────────────────────

def build_message(df: pd.DataFrame) -> str:
    today    = datetime.today().strftime("%d %b %Y")
    total    = len(df)
    grade_a  = df[df["grade"] == "A"] if "grade" in df.columns else df.head(0)
    clean_df = df[df["is_clean"]] if "is_clean" in df.columns else df

    lines = []

    # ── Header ─────────────────────────────────────────────────────
    lines.append(f"📈 <b>RS SCREENER — {today}</b>")
    lines.append(f"Universe: {total} stocks screened\n")

    # ── Top 5 RS Leaders ───────────────────────────────────────────
    lines.append("🏆 <b>TOP 5 RS LEADERS</b>")
    top5 = clean_df.head(5)
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        rs_sign = "+" if row["rs"] >= 0 else ""
        lines.append(
            f"{i}. <b>{row['symbol']}</b>  "
            f"₹{row['close']:,.0f}  "
            f"RS: {rs_sign}{row['rs']:.4f}  "
            f"[{row.get('sector','—')}]"
        )

    lines.append("")

    # ── Grade A stocks ─────────────────────────────────────────────
    lines.append(f"🟢 <b>GRADE A — TOP LEADERS ({len(grade_a)} stocks)</b>")

    if grade_a.empty:
        lines.append("No Grade A stocks this week.")
    else:
        # Group by sector for cleaner display
        for sector, group in grade_a.groupby("sector"):
            symbols = ", ".join(group["symbol"].tolist())
            lines.append(f"  <b>{sector}</b>: {symbols}")

    lines.append("")

    # ── Sector breakdown ───────────────────────────────────────────
    lines.append("🏭 <b>SECTOR BREAKDOWN (Grade A + B)</b>")
    top_stocks  = df[df["grade"].isin(["A","B"])] if "grade" in df.columns else df
    sector_dist = (
        top_stocks.groupby("sector")
        .agg(Count=("symbol","count"), Avg_RS=("rs","mean"))
        .sort_values("Count", ascending=False)
        .head(10)
    )
    for sector, row in sector_dist.iterrows():
        bar = "▓" * min(int(row["Count"] / 2), 20)
        lines.append(
            f"  {sector:<14} {int(row['Count']):>3}  "
            f"Avg RS: {row['Avg_RS']:+.3f}  {bar}"
        )

    lines.append("")
    lines.append(f"⏰ Generated: {datetime.today().strftime('%d %b %Y %H:%M')}")
    lines.append("─────────────────────────")
    lines.append("RS Screener | Weekly | NSE Universe")

    return "\n".join(lines)


# ── Split long messages ────────────────────────────────────────────

def send_results(df: pd.DataFrame) -> bool:
    """
    Build and send the screener results message.
    Splits into multiple messages if too long (Telegram limit: 4096 chars).
    """
    if not BOT_TOKEN or not CHAT_ID:
        print("  ❌ BOT_TOKEN or CHAT_ID not set in .env")
        return False

    message = build_message(df)

    # Split if over Telegram's 4096 char limit
    if len(message) <= 4096:
        ok = send_message(message)
    else:
        chunks = []
        current = ""
        for line in message.split("\n"):
            if len(current) + len(line) + 1 > 4000:
                chunks.append(current)
                current = line
            else:
                current += "\n" + line
        if current:
            chunks.append(current)

        ok = True
        for chunk in chunks:
            if not send_message(chunk):
                ok = False

    return ok


# ── Verification test ──────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("  TELEGRAM — VERIFICATION TEST")
    print("=" * 55)

    if not BOT_TOKEN or not CHAT_ID:
        print("  ❌ Credentials not found in .env")
        print("  Make sure .env has TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        exit()

    print(f"  Bot Token : ...{BOT_TOKEN[-8:]}")
    print(f"  Chat ID   : {CHAT_ID}")

    # Load data
    df = load_latest_scored()
    if df is None:
        print("  No scored CSV found. Sending test message instead.")
        ok = send_message("✅ RS Screener bot is working! Test message.")
    else:
        print(f"  Loaded    : {len(df)} stocks")

        # Score if needed
        if "grade" not in df.columns:
            from core.scorer import clean_and_score
            df = clean_and_score(df)

        print("  Building message...")
        msg = build_message(df)
        print(f"  Message length: {len(msg)} chars")
        print("\n--- MESSAGE PREVIEW ---")
        print(msg[:500] + "..." if len(msg) > 500 else msg)
        print("--- END PREVIEW ---\n")

        print("  Sending to Telegram...")
        ok = send_results(df)

    if ok:
        print("  ✅ Message sent! Check your Telegram.")
    else:
        print("  ❌ Send failed. Check internet / token.")
