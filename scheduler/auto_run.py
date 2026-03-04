"""
auto_run.py
-----------
Weekly scheduler for the RS Screener.
Runs every Saturday at 8:00 AM automatically.

What it does each Saturday:
  1. Runs full screener (2264 stocks)
  2. Scores + grades results
  3. Exports to Excel
  4. Sends summary to Telegram
  5. Pushes scored CSV to GitHub (so web app shows fresh results)

Usage:
  python3 scheduler/auto_run.py          # start scheduler (runs in background)
  python3 scheduler/auto_run.py --now    # trigger one run immediately (for testing)
"""

import os, sys, glob, argparse, logging, subprocess
import pandas as pd
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR  = os.path.join(BASE_DIR, "logs")
sys.path.insert(0, BASE_DIR)
os.makedirs(LOG_DIR, exist_ok=True)

from core.screener          import run_screener
from core.scorer            import clean_and_score
from output.excel_exporter  import export_to_excel
from output.telegram        import send_results, send_message
from core.conditions        import DEFAULT_CONDITIONS

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))


# ── Logger ─────────────────────────────────────────────────────────

def setup_logger() -> logging.Logger:
    log_file = os.path.join(LOG_DIR, "auto_run.log")
    logger   = logging.getLogger("auto_run")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        ))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(
            "%(asctime)s | %(message)s"
        ))
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# ── Git push results ───────────────────────────────────────────────

def push_results_to_github(scored_path: str, run_id: str,
                           logger: logging.Logger):
    """
    Push the scored CSV to GitHub so Streamlit Cloud shows fresh results.
    Steps:
      1. Force-add the scored CSV (bypasses .gitignore)
      2. Remove older CSVs from git (keep latest 4 only)
      3. Commit with run date
      4. Push to origin main
    """
    try:
        # Force-add the scored CSV (it's in .gitignore so we use -f)
        subprocess.run(
            ["git", "add", "-f", scored_path],
            cwd=BASE_DIR, check=True, capture_output=True
        )

        # Remove older scored CSVs from git (keep only latest 4)
        results_dir = os.path.join(BASE_DIR, "results", "manual")
        old_csvs = sorted(
            glob.glob(os.path.join(results_dir, "scored_*.csv")),
            reverse=True
        )[4:]
        for old in old_csvs:
            subprocess.run(
                ["git", "rm", "--cached", old],
                cwd=BASE_DIR, capture_output=True
            )

        # Commit
        commit_msg = f"Results update — {datetime.today().strftime('%d %b %Y')}"
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=BASE_DIR, capture_output=True, text=True
        )

        if "nothing to commit" in result.stdout:
            logger.info("GitHub: nothing new to commit")
            return

        # Push
        push = subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=BASE_DIR, capture_output=True, text=True
        )

        if push.returncode == 0:
            logger.info("GitHub: results pushed successfully")
            logger.info("Streamlit Cloud will update within 1 minute")
        else:
            logger.warning(f"GitHub push failed: {push.stderr}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {e}")
    except Exception as e:
        logger.error(f"GitHub push error: {e}")


# ── Full weekly run ────────────────────────────────────────────────

def weekly_run():
    """
    Complete weekly pipeline:
    screener -> scorer -> excel -> telegram -> github push
    """
    logger = setup_logger()
    run_id = datetime.today().strftime("%Y%m%d_%H%M")

    logger.info("=" * 55)
    logger.info(f"WEEKLY RUN STARTED  |  Run ID: {run_id}")
    logger.info("=" * 55)

    send_message(f"⏳ <b>RS Screener starting...</b>\nRun ID: {run_id}")

    # Step 1: Screener
    logger.info("Step 1/5 — Running screener (2264 stocks)")
    try:
        results_df = run_screener(
            benchmark   = "^CNX200",
            conditions  = DEFAULT_CONDITIONS,
            rs_period   = 52,
            ma_period   = 20,
            sma_period  = 20,
            run_id      = run_id,
            delay_secs  = 0.3,
        )
        logger.info(f"Screener done — {len(results_df)} stocks processed")
    except Exception as e:
        logger.error(f"Screener failed: {e}")
        send_message(f"❌ <b>Screener failed</b>\nError: {e}")
        return

    if results_df.empty:
        logger.error("Screener returned empty results")
        send_message("❌ <b>Screener returned no results</b>")
        return

    # Step 2: Score
    logger.info("Step 2/5 — Scoring + grading results")
    try:
        scored_df = clean_and_score(results_df)
        grade_a   = len(scored_df[scored_df["grade"] == "A"])
        grade_b   = len(scored_df[scored_df["grade"] == "B"])
        logger.info(f"Scoring done — Grade A: {grade_a} | Grade B: {grade_b}")

        scored_path = os.path.join(
            BASE_DIR, "results", "manual", f"scored_{run_id}.csv"
        )
        scored_df.to_csv(scored_path, index=False)
        logger.info(f"Scored CSV saved: scored_{run_id}.csv")
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        send_message(f"❌ <b>Scoring failed</b>\nError: {e}")
        return

    # Step 3: Excel
    logger.info("Step 3/5 — Exporting to Excel")
    try:
        excel_path = export_to_excel(scored_df)
        logger.info(f"Excel saved: {os.path.basename(excel_path)}")
    except Exception as e:
        logger.error(f"Excel export failed: {e}")

    # Step 4: Telegram
    logger.info("Step 4/5 — Sending Telegram summary")
    try:
        ok = send_results(scored_df)
        if ok:
            logger.info("Telegram message sent successfully")
        else:
            logger.warning("Telegram send failed")
    except Exception as e:
        logger.error(f"Telegram failed: {e}")

    # Step 5: Push to GitHub
    logger.info("Step 5/5 — Pushing results to GitHub")
    try:
        push_results_to_github(scored_path, run_id, logger)
    except Exception as e:
        logger.error(f"GitHub push failed: {e}")

    elapsed_note = datetime.today().strftime("%d %b %Y %H:%M")
    logger.info(f"WEEKLY RUN COMPLETE  |  {elapsed_note}")
    logger.info("=" * 55)


# ── Scheduler setup ────────────────────────────────────────────────

def start_scheduler():
    logger = setup_logger()

    scheduler = BlockingScheduler(timezone="Asia/Kolkata")

    scheduler.add_job(
        func    = weekly_run,
        trigger = CronTrigger(
            day_of_week = "sat",
            hour        = 8,
            minute      = 0,
            timezone    = "Asia/Kolkata",
        ),
        id               = "weekly_rs_screener",
        name             = "Weekly RS Screener",
        replace_existing = True,
    )

    job = scheduler.get_jobs()[0]
    logger.info("=" * 55)
    logger.info("  RS SCREENER SCHEDULER STARTED")
    logger.info("  Schedule : Every Saturday at 8:00 AM IST")
    logger.info(f"  Job      : {job.name}")
    logger.info("  Press Ctrl+C to stop")
    logger.info("=" * 55)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user.")


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RS Screener Scheduler")
    parser.add_argument(
        "--now",
        action = "store_true",
        help   = "Run the screener immediately (skip schedule)"
    )
    args = parser.parse_args()

    if args.now:
        print("=" * 55)
        print("  MANUAL RUN TRIGGERED (--now flag)")
        print("=" * 55)
        weekly_run()
    else:
        start_scheduler()
