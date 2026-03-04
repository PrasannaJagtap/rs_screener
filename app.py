"""
app.py
------
Streamlit UI for the RS Screener.

Sections:
  1. Sidebar    — run controls, condition toggles, filters
  2. Run tab    — trigger screener, live log output
  3. Results tab — Grade A/B/C/D table with filters
  4. Sectors tab — sector breakdown chart
  5. Stock tab   — individual stock RS chart + conditions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys, subprocess, glob, json
from datetime import datetime
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "manual")
LOG_DIR     = os.path.join(BASE_DIR, "logs")
CORE_DIR    = os.path.join(BASE_DIR, "core")
sys.path.insert(0, BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

from core.data_fetcher       import fetch_stock, fetch_benchmark, align_data
from core.crossover_scanner  import run_crossover_scanner
from core.rs_calculator import build_rs_summary
from core.conditions    import DEFAULT_CONDITIONS
from core.scorer        import clean_and_score

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title = "RS Screener",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Password gate ──────────────────────────────────────────────────
def get_app_password() -> str:
    try:
        return st.secrets["APP_PASSWORD"]
    except Exception:
        return os.getenv("APP_PASSWORD", "")

APP_PASSWORD = get_app_password()

if APP_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("## 📈 RS Screener")
        st.markdown("Enter the password to continue.")
        pwd = st.text_input("Password", type="password")
        if st.button("Login", type="primary"):
            if pwd == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password. Try again.")
        st.stop()

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark theme refinements */
  .stApp { background-color: #0e1117; }

  /* Grade badges */
  .badge-A { background:#16a34a; color:#fff; padding:2px 10px;
              border-radius:4px; font-weight:700; font-size:13px; }
  .badge-B { background:#2563eb; color:#fff; padding:2px 10px;
              border-radius:4px; font-weight:700; font-size:13px; }
  .badge-C { background:#d97706; color:#fff; padding:2px 10px;
              border-radius:4px; font-weight:700; font-size:13px; }
  .badge-D { background:#dc2626; color:#fff; padding:2px 10px;
              border-radius:4px; font-weight:700; font-size:13px; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #1c1f26;
    border: 1px solid #2d3139;
    border-radius: 8px;
    padding: 12px 16px;
  }

  /* Table styling */
  .dataframe { font-size: 13px !important; }

  /* Section headers */
  .section-header {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def list_scored_csvs() -> list[str]:
    files = glob.glob(os.path.join(RESULTS_DIR, "scored_*.csv"))
    files.sort(reverse=True)
    return files


def list_screener_csvs() -> list[str]:
    files = glob.glob(os.path.join(RESULTS_DIR, "screener_*.csv"))
    files.sort(reverse=True)
    return files


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Parse conditions column if stored as JSON string
    if "conditions" in df.columns and isinstance(df["conditions"].iloc[0], str):
        try:
            df["conditions"] = df["conditions"].apply(json.loads)
        except Exception:
            pass
    return df


def grade_color(g: str) -> str:
    return {"A": "#16a34a", "B": "#2563eb", "C": "#d97706", "D": "#dc2626"}.get(g, "#6b7280")


def format_rs(val: float) -> str:
    return f"+{val:.4f}" if val >= 0 else f"{val:.4f}"


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📈 RS Screener")
    st.markdown("---")

    # ── Condition toggles ─────────────────────────────────────────
    st.markdown('<p class="section-header">Active Conditions</p>', unsafe_allow_html=True)

    condition_labels = {
        "C1_RS_Positive"          : "C1 — RS > 0",
        "C2_RS_Slope_Positive"    : "C2 — RS Slope Rising",
        "C3_RS_MA_Rising"         : "C3 — RS MA Rising (3 bars)",
        "C4_Price_Above_SMA"      : "C4 — Price > SMA 20",
        "C5_Above_WMA200"         : "C5 — Price > WMA 200",
        "C6_RS_Crossed_Above_MA"  : "C6 — RS Crossed above MA",
        "C7_RS_Depth_Limit"       : "C7 — RS Depth Limit (>-10%)",
        "C8_Former_Leader"        : "C8 — Former Leader",
        "C9_Crossover_Sustained"  : "C9 — Crossover Sustained",
        "C10_Min_Volume"          : "C10 — Min Volume",
        "C11_Near_52w_High"       : "C11 — Near 52-week High",
        "C12_RS_Zero_Cross_Recent": "C12 — RS Crossed Zero Recently",
    }

    selected_conditions = {}
    for key, label in condition_labels.items():
        selected_conditions[key] = st.checkbox(
            label,
            value = DEFAULT_CONDITIONS.get(key, False),
            key   = f"cond_{key}",
        )

    st.markdown("---")

    # ── Screener settings ─────────────────────────────────────────
    st.markdown('<p class="section-header">Screener Settings</p>', unsafe_allow_html=True)

    benchmark  = st.selectbox("Benchmark", ["^CNX200", "^NSEI", "^NSEBANK"], index=0)
    rs_period  = st.slider("RS Period (weeks)", 26, 104, 52, 4)
    ma_period  = st.slider("RS MA Period (weeks)", 5, 50, 20, 5)
    min_score  = st.slider("Min Score Threshold", 1, 12, 4, 1)

    st.markdown("---")

    # ── Run button ────────────────────────────────────────────────
    st.markdown('<p class="section-header">Run Screener</p>', unsafe_allow_html=True)
    run_button = st.button("🚀 Run Full Screener", type="primary", use_container_width=True)
    st.caption("Takes ~25 min for full NSE universe")

    st.markdown("---")
    st.markdown('<p class="section-header">Crossover Scanner</p>', unsafe_allow_html=True)
    cross_lookback = st.selectbox("Lookback (weeks)", [1, 2, 3, 4, 5], index=2, key="cross_lookback")
    cross_min_rs   = st.slider("Min RS at crossover", -1.0, 0.5, -0.5, 0.1, key="cross_min_rs")
    cross_top_n    = st.selectbox("Top N picks", [10, 15, 20, 30], index=2, key="cross_top_n")
    cross_button   = st.button("🔀 Run Crossover Scanner", use_container_width=True, key="cross_btn")
    st.caption("Takes ~25 min for full NSE universe")


# ══════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════

tab_results, tab_run, tab_sectors, tab_stock, tab_cross = st.tabs([
    "📊 Results", "⚙️ Run Screener", "🏭 Sectors", "🔍 Stock Detail", "🔀 Crossover"
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — RESULTS
# ══════════════════════════════════════════════════════════════════

with tab_results:

    st.markdown("### Results")

    # File picker
    scored_files = list_scored_csvs()
    screener_files = list_screener_csvs()
    all_files = scored_files + screener_files

    if not all_files:
        st.warning("No results found. Run the screener first.")
        st.stop()

    def friendly_name(path):
        name = os.path.basename(path)
        prefix = "scored_" if "scored_" in name else "screener_"
        ts = name.replace(prefix, "").replace(".csv", "")
        try:
            dt = datetime.strptime(ts, "%Y%m%d_%H%M")
            tag = "✅ Scored" if "scored_" in name else "📄 Raw"
            return f"{tag}  {dt.strftime('%d %b %Y  %H:%M')}"
        except Exception:
            return name

    file_labels = [friendly_name(f) for f in all_files]
    selected_idx = st.selectbox(
        "Select run",
        range(len(all_files)),
        format_func=lambda i: file_labels[i],
        index=0,
    )
    selected_file = all_files[selected_idx]

    raw_df = load_csv(selected_file)

    # Score if raw
    if "grade" not in raw_df.columns:
        with st.spinner("Scoring results..."):
            df = clean_and_score(raw_df)
    else:
        df = raw_df.copy()

    # ── Summary metrics ───────────────────────────────────────────
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    total   = len(df[df.get("is_clean", pd.Series([True]*len(df)))])
    grade_a = len(df[df["grade"] == "A"]) if "grade" in df.columns else 0
    grade_b = len(df[df["grade"] == "B"]) if "grade" in df.columns else 0
    grade_c = len(df[df["grade"] == "C"]) if "grade" in df.columns else 0
    grade_d = len(df[df["grade"] == "D"]) if "grade" in df.columns else 0
    flagged = len(df[~df["is_clean"]]) if "is_clean" in df.columns else 0

    col1.metric("Total", f"{len(df):,}")
    col2.metric("Grade A 🟢", grade_a)
    col3.metric("Grade B 🔵", grade_b)
    col4.metric("Grade C 🟡", grade_c)
    col5.metric("Grade D 🔴", grade_d)
    col6.metric("Flagged ⚠️", flagged)

    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        grade_filter = st.multiselect(
            "Grade", ["A", "B", "C", "D"],
            default=["A", "B"]
        )
    with f2:
        tier_filter = st.multiselect(
            "Tier",
            ["Tier 1", "Tier 2", "Watchlist", "Filtered"],
            default=["Tier 1", "Tier 2"]
        )
    with f3:
        sectors = sorted(df["sector"].unique()) if "sector" in df.columns else []
        sector_filter = st.multiselect("Sector", sectors, default=[])

    with f4:
        clean_only = st.checkbox("Clean data only", value=True)

    # Apply filters
    filtered = df.copy()
    if grade_filter and "grade" in filtered.columns:
        filtered = filtered[filtered["grade"].isin(grade_filter)]
    if tier_filter and "tier" in filtered.columns:
        filtered = filtered[filtered["tier"].isin(tier_filter)]
    if sector_filter and "sector" in filtered.columns:
        filtered = filtered[filtered["sector"].isin(sector_filter)]
    if clean_only and "is_clean" in filtered.columns:
        filtered = filtered[filtered["is_clean"]]

    st.markdown(f"**{len(filtered)} stocks** matching filters")

    # ── Display table ─────────────────────────────────────────────
    display_cols = ["symbol", "grade", "rs_rank", "rs", "close",
                    "tier", "score", "total_active", "sector"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    show = filtered[display_cols].copy()

    # Format
    if "rs" in show.columns:
        show["rs"] = show["rs"].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "")
    if "rs_rank" in show.columns:
        show["rs_rank"] = show["rs_rank"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
    if "close" in show.columns:
        show["close"] = show["close"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "")
    if "score" in show.columns and "total_active" in show.columns:
        show["score"] = show.apply(lambda r: f"{int(r['score'])}/{int(r['total_active'])}", axis=1)
        show = show.drop(columns=["total_active"])

    show.columns = [c.replace("_", " ").title() for c in show.columns]
    st.dataframe(show, use_container_width=True, height=500)

    # ── Download ──────────────────────────────────────────────────
    csv_out = filtered.to_csv(index=False)
    st.download_button(
        "⬇️ Download filtered results (CSV)",
        data     = csv_out,
        file_name= f"rs_filtered_{datetime.today().strftime('%Y%m%d_%H%M')}.csv",
        mime     = "text/csv",
    )


# ══════════════════════════════════════════════════════════════════
# TAB 2 — RUN SCREENER
# ══════════════════════════════════════════════════════════════════

with tab_run:
    st.markdown("### Run Screener")

    active_count = sum(1 for v in selected_conditions.values() if v)
    st.info(f"**{active_count} conditions active** — {benchmark} benchmark — RS Period: {rs_period}w — MA: {ma_period}w")

    # Show active conditions
    cond_cols = st.columns(3)
    for i, (key, label) in enumerate(condition_labels.items()):
        is_active = selected_conditions.get(key, False)
        icon = "✅" if is_active else "⬜"
        cond_cols[i % 3].markdown(f"{icon} {label}")

    st.markdown("---")

    if run_button:
        st.markdown("#### Screener Running...")
        progress_bar = st.progress(0)
        status_text  = st.empty()
        log_area     = st.empty()

        # Write a temp config for the screener to pick up
        config = {
            "benchmark"         : benchmark,
            "rs_period"         : rs_period,
            "ma_period"         : ma_period,
            "min_threshold"     : min_score,
            "active_conditions" : selected_conditions,
        }
        config_path = os.path.join(BASE_DIR, ".screener_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        run_id = datetime.today().strftime("%Y%m%d_%H%M")
        log_file = os.path.join(LOG_DIR, f"screener_{run_id}.log")
        os.makedirs(LOG_DIR, exist_ok=True)

        cmd = [
            sys.executable, "-c",
            f"""
import sys, json
sys.path.insert(0, '{BASE_DIR}')
from core.screener import run_screener

with open('{config_path}') as f:
    cfg = json.load(f)

run_screener(
    benchmark   = cfg['benchmark'],
    rs_period   = cfg['rs_period'],
    ma_period   = cfg['ma_period'],
    run_id      = '{run_id}',
    conditions  = cfg['active_conditions'],
    delay_secs  = 0.3,
)
"""
        ]

        status_text.markdown("⏳ Fetching stock universe and running calculations...")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            log_lines = []
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    log_lines.append(line)
                    # Update progress from progress bar info in line
                    if "%" in line and "/" in line:
                        try:
                            pct_str = line.split("(")[1].split("%")[0]
                            pct = min(float(pct_str) / 100, 1.0)
                            progress_bar.progress(pct)
                        except Exception:
                            pass
                    # Show last 20 log lines
                    log_area.code("\n".join(log_lines[-20:]), language=None)

            proc.wait()
            progress_bar.progress(1.0)

            if proc.returncode == 0:
                status_text.success(f"✅ Screener complete! Run ID: {run_id}")

                # Auto-score the result
                raw_path = os.path.join(RESULTS_DIR, f"screener_{run_id}.csv")
                if os.path.exists(raw_path):
                    with st.spinner("Scoring results..."):
                        raw = pd.read_csv(raw_path)
                        scored = clean_and_score(raw)
                        scored_path = os.path.join(
                            RESULTS_DIR, f"scored_{run_id}.csv"
                        )
                        scored.to_csv(scored_path, index=False)

                    t1 = len(scored[scored["tier"] == "Tier 1"])
                    t2 = len(scored[scored["tier"] == "Tier 2"])
                    ga = len(scored[scored["grade"] == "A"])

                    st.success(f"Scored & saved. Grade A: **{ga}** | Tier 1: **{t1}** | Tier 2: **{t2}**")
                    st.info("Switch to the **Results** tab and select this run from the dropdown.")
            else:
                status_text.error("❌ Screener failed. Check log output above.")

        except Exception as e:
            status_text.error(f"Error: {e}")

    else:
        st.markdown("""
**Instructions:**
1. Toggle conditions in the sidebar
2. Adjust settings (benchmark, RS period, MA period)
3. Click **🚀 Run Full Screener**
4. Wait ~25 minutes (progress shown live)
5. Results auto-scored and available in **Results** tab
        """)

        # Show past run log if available
        log_files = sorted(glob.glob(os.path.join(LOG_DIR, "screener_*.log")), reverse=True)
        if log_files:
            latest_log = log_files[0]
            log_name   = os.path.basename(latest_log)
            with st.expander(f"Last run log: {log_name}"):
                with open(latest_log) as f:
                    lines = f.readlines()[-50:]
                st.code("".join(lines), language=None)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — SECTORS
# ══════════════════════════════════════════════════════════════════

with tab_sectors:
    st.markdown("### Sector Breakdown")

    if "df" not in dir() or df is None or "sector" not in df.columns:
        st.info("Load results in the Results tab first.")
    else:
        grade_sel = st.multiselect(
            "Show grades",
            ["A", "B", "C", "D"],
            default=["A", "B"],
            key="sector_grade"
        )

        sector_df = df[df["grade"].isin(grade_sel)] if "grade" in df.columns else df

        if sector_df.empty:
            st.warning("No stocks match selected grades.")
        else:
            # ── Sector count bar chart ─────────────────────────────
            sc = sector_df["sector"].value_counts().reset_index()
            sc.columns = ["Sector", "Count"]

            fig_bar = px.bar(
                sc,
                x       = "Count",
                y       = "Sector",
                orientation = "h",
                color   = "Count",
                color_continuous_scale = "Teal",
                title   = f"Stocks per Sector (Grade {', '.join(grade_sel)})",
            )
            fig_bar.update_layout(
                template        = "plotly_dark",
                paper_bgcolor   = "#0e1117",
                plot_bgcolor    = "#0e1117",
                showlegend      = False,
                coloraxis_showscale = False,
                margin          = dict(l=20, r=20, t=40, b=20),
                height          = 500,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Average RS by sector ───────────────────────────────
            avg_rs = (
                sector_df.groupby("sector")["rs"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            avg_rs.columns = ["Sector", "Avg RS"]

            fig_rs = px.bar(
                avg_rs,
                x     = "Avg RS",
                y     = "Sector",
                orientation = "h",
                color = "Avg RS",
                color_continuous_scale = "RdYlGn",
                title = "Average RS by Sector",
            )
            fig_rs.update_layout(
                template        = "plotly_dark",
                paper_bgcolor   = "#0e1117",
                plot_bgcolor    = "#0e1117",
                showlegend      = False,
                coloraxis_showscale = False,
                margin          = dict(l=20, r=20, t=40, b=20),
                height          = 500,
            )
            st.plotly_chart(fig_rs, use_container_width=True)

            # ── Sector detail table ────────────────────────────────
            with st.expander("Sector detail table"):
                sector_detail = (
                    sector_df.groupby("sector")
                    .agg(
                        Count    = ("symbol", "count"),
                        Avg_RS   = ("rs", "mean"),
                        Max_RS   = ("rs", "max"),
                        Tier1    = ("tier", lambda x: (x == "Tier 1").sum()),
                    )
                    .sort_values("Count", ascending=False)
                    .reset_index()
                )
                sector_detail["Avg_RS"] = sector_detail["Avg_RS"].round(4)
                sector_detail["Max_RS"] = sector_detail["Max_RS"].round(4)
                st.dataframe(sector_detail, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — STOCK DETAIL
# ══════════════════════════════════════════════════════════════════

with tab_stock:
    st.markdown("### Stock Detail")

    col_inp, col_cfg = st.columns([2, 1])

    with col_inp:
        stock_input = st.text_input(
            "Enter NSE symbol",
            placeholder = "e.g. RELIANCE, SBIN, HINDCOPPER",
            value       = "",
        ).strip().upper()

    with col_cfg:
        detail_period = st.selectbox(
            "RS Period", [26, 52, 78, 104], index=1,
            format_func=lambda x: f"{x} weeks ({x//52}y {x%52}w)" if x >= 52 else f"{x} weeks"
        )

    if stock_input:
        with st.spinner(f"Fetching {stock_input}..."):
            bench_df = fetch_benchmark("^CNX200")
            stock_df = fetch_stock(stock_input)

        if bench_df is None or stock_df is None:
            st.error(f"Could not fetch data for **{stock_input}**. Check symbol.")
        else:
            s_aligned, b_aligned = align_data(stock_df, bench_df)
            summary = build_rs_summary(
                stock_close  = s_aligned["Close"],
                bench_close  = b_aligned["Close"],
                rs_period    = detail_period,
                ma_period    = 20,
                sma_period   = 20,
                slope_period = 5,
            )

            from core.conditions import score_stock
            report = score_stock(
                summary       = summary,
                stock_df      = stock_df,
                conditions    = selected_conditions,
                min_threshold = min_score,
            )

            latest = summary.iloc[-1]
            conds  = report["conditions"]

            # ── Header metrics ─────────────────────────────────────
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Close",    f"₹{latest['Close']:,.2f}")
            m2.metric("RS",       f"{latest['RS']:+.4f}")
            m3.metric("RS MA",    f"{latest['RS_MA']:.4f}")
            m4.metric("Score",    f"{report['score']}/{report['total_active']}")
            m5.metric("Tier",     report["tier"])

            st.markdown("---")

            chart_col, cond_col = st.columns([3, 1])

            # ── RS Chart ───────────────────────────────────────────
            with chart_col:
                fig = go.Figure()

                # Price in top panel
                fig.add_trace(go.Scatter(
                    x    = summary.index,
                    y    = summary["Close"],
                    name = f"{stock_input} Price",
                    line = dict(color="#60a5fa", width=1.5),
                    yaxis = "y1",
                ))
                fig.add_trace(go.Scatter(
                    x    = summary.index,
                    y    = summary["Price_SMA_20"],
                    name = "SMA 20",
                    line = dict(color="#f59e0b", width=1, dash="dot"),
                    yaxis = "y1",
                ))
                if "Price_WMA_200" in summary.columns:
                    fig.add_trace(go.Scatter(
                        x    = summary.index,
                        y    = summary["Price_WMA_200"],
                        name = "WMA 200",
                        line = dict(color="#a78bfa", width=1, dash="dash"),
                        yaxis = "y1",
                    ))

                # RS line in bottom panel
                rs_colors = ["#ef4444" if v < 0 else "#22c55e" for v in summary["RS"]]
                fig.add_trace(go.Bar(
                    x     = summary.index,
                    y     = summary["RS"],
                    name  = "RS",
                    marker_color = rs_colors,
                    yaxis = "y2",
                    opacity = 0.7,
                ))
                fig.add_trace(go.Scatter(
                    x    = summary.index,
                    y    = summary["RS_MA"],
                    name = "RS MA",
                    line = dict(color="#f59e0b", width=1.5),
                    yaxis = "y2",
                ))
                # Zero line
                fig.add_hline(
                    y         = 0,
                    line_dash = "dot",
                    line_color= "#6b7280",
                    yref      = "y2",
                )

                fig.update_layout(
                    template     = "plotly_dark",
                    paper_bgcolor= "#0e1117",
                    plot_bgcolor = "#0e1117",
                    title        = f"{stock_input} — Price & RS ({detail_period}w)",
                    height       = 550,
                    hovermode    = "x unified",
                    legend       = dict(orientation="h", y=1.08),
                    margin       = dict(l=10, r=10, t=60, b=10),
                    yaxis  = dict(domain=[0.35, 1.0], title="Price ₹", gridcolor="#1f2937"),
                    yaxis2 = dict(domain=[0.0, 0.30], title="RS",      gridcolor="#1f2937"),
                    xaxis  = dict(gridcolor="#1f2937"),
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── Conditions checklist ───────────────────────────────
            with cond_col:
                st.markdown("**Conditions**")
                for key, label in condition_labels.items():
                    if not selected_conditions.get(key, False):
                        continue
                    passed = conds.get(key, False)
                    icon   = "✅" if passed else "❌"
                    short  = label.split("—")[1].strip() if "—" in label else label
                    c_key  = label.split("—")[0].strip()
                    st.markdown(f"{icon} **{c_key}** {short}")

                st.markdown("---")
                st.markdown(f"**Score:** {report['score']}/{report['total_active']}")
                st.markdown(f"**Tier:** {report['tier']}")

                rs_val = latest["RS"]
                if rs_val > 0:
                    st.success(f"RS: {rs_val:+.4f}")
                else:
                    st.error(f"RS: {rs_val:+.4f}")

                slope = latest["RS_Slope"]
                if slope > 0:
                    st.success("Slope: Rising ↑")
                else:
                    st.warning("Slope: Falling ↓")

    else:
        st.info("Enter an NSE symbol above to see RS chart and condition breakdown.")
        st.markdown("""
**Examples to try:**
- `HINDCOPPER` — Grade A, RS +1.42
- `SBIN` — Grade A, Banking leader
- `RELIANCE` — RS near zero, pullback mode
- `INFY` — RS negative, IT sector weak
        """)


# ══════════════════════════════════════════════════════════════════
# TAB 5 — CROSSOVER SCANNER
# ══════════════════════════════════════════════════════════════════

with tab_cross:
    st.markdown("### 🔀 Crossover Scanner")
    st.markdown("Finds stocks where **RS crossed above RS MA** recently — early momentum entry signal.")

    st.markdown("---")

    # ── Settings summary ──────────────────────────────────────────
    st.info(
        f"**Settings:** Lookback {cross_lookback} week{'s' if cross_lookback > 1 else ''} "
        f"| Min RS at crossover: {cross_min_rs:+.1f} "
        f"| Top picks: {cross_top_n} "
        f"| Benchmark: {benchmark}"
    )

    if cross_button:

        # ── Progress placeholders ──────────────────────────────────
        cross_progress = st.progress(0)
        cross_status   = st.empty()
        cross_log      = st.empty()

        cross_status.markdown("⏳ Fetching benchmark and scanning NSE universe...")

        try:
            # Run in-process but show a spinner
            # We capture stdout via a subprocess so the progress bar can update
            import tempfile, threading, queue

            config_cross = {
                "lookback"  : cross_lookback,
                "min_rs"    : cross_min_rs,
                "benchmark" : benchmark,
                "rs_period" : rs_period,
                "ma_period" : ma_period,
            }
            cross_config_path = os.path.join(BASE_DIR, ".cross_config.json")
            cross_run_id      = datetime.today().strftime("%Y%m%d_%H%M")
            cross_out_path    = os.path.join(
                RESULTS_DIR, f"crossover_{cross_lookback}w_{cross_run_id}.csv"
            )
            cross_picks_path  = os.path.join(
                RESULTS_DIR, f"top_picks_{cross_lookback}w_{cross_run_id}.csv"
            )

            with open(cross_config_path, "w") as f:
                json.dump(config_cross, f)

            cmd_cross = [
                sys.executable, "-c",
                f"""
import sys, json
sys.path.insert(0, '{BASE_DIR}')
from core.crossover_scanner import run_crossover_scanner, save_results, save_top_picks, print_results

with open('{cross_config_path}') as f:
    cfg = json.load(f)

df = run_crossover_scanner(
    lookback   = cfg['lookback'],
    min_rs     = cfg['min_rs'],
    benchmark  = cfg['benchmark'],
    rs_period  = cfg['rs_period'],
    ma_period  = cfg['ma_period'],
)
if not df.empty:
    df.to_csv('{cross_out_path}', index=False)

    # Top picks
    picks = df[
        (df['sustained'] == True) &
        (df['rs_now']    >  0.0)  &
        (df['rs_gap']    >  0.02) &
        (df['rs_slope']  >  0.0)
    ].copy()
    if not picks.empty:
        picks['score'] = picks['rs_now']*0.4 + picks['rs_gap']*0.4 + picks['rs_slope']*0.2
        picks = picks.sort_values('score', ascending=False).head({cross_top_n})

        def conviction(row):
            if row['rs_now'] > 0.5 and row['rs_gap'] > 0.1 and row['rs_slope'] > 0.1:
                return 'HIGH'
            elif row['rs_now'] > 0.2 and row['rs_gap'] > 0.05:
                return 'MED'
            else:
                return 'WATCH'
        picks['conviction'] = picks.apply(conviction, axis=1)
        picks.to_csv('{cross_picks_path}', index=False)
"""
            ]

            log_lines_cross = []
            proc_cross = subprocess.Popen(
                cmd_cross,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in proc_cross.stdout:
                line = line.rstrip()
                if not line:
                    continue
                log_lines_cross.append(line)
                # Update progress
                if "%" in line and "/" in line:
                    try:
                        pct_str = line.split("(")[1].split("%")[0]
                        pct = min(float(pct_str) / 100, 1.0)
                        cross_progress.progress(pct)
                    except Exception:
                        pass
                cross_log.code("\n".join(log_lines_cross[-15:]), language=None)

            proc_cross.wait()
            cross_progress.progress(1.0)

            # ── Load and display results ───────────────────────────
            if os.path.exists(cross_out_path):
                cross_df    = pd.read_csv(cross_out_path)
                cross_status.success(f"✅ Scan complete — {len(cross_df)} crossovers found")

                # Load top picks if saved
                picks_df = pd.DataFrame()
                if os.path.exists(cross_picks_path):
                    picks_df = pd.read_csv(cross_picks_path)

                st.markdown("---")

                # ── Summary metrics ────────────────────────────────
                m1, m2, m3, m4 = st.columns(4)
                sustained_n = len(cross_df[cross_df["sustained"] == True])
                fresh_n     = len(cross_df[cross_df["sustained"] == False])
                m1.metric("Total Crossovers", len(cross_df))
                m2.metric("✅ Sustained",     sustained_n)
                m3.metric("⏳ Fresh Only",    fresh_n)
                m4.metric("🏆 Top Picks",     len(picks_df))

                st.markdown("---")

                # ══ TOP PICKS SECTION ════════════════════════════
                st.markdown("## 🏆 Top Picks")
                st.caption("Filtered: Sustained ✅ | RS > 0 | Gap > 0.02 | Slope rising | Ranked by composite score")

                if picks_df.empty:
                    st.warning("No stocks passed all quality filters. Try relaxing Min RS or increasing Lookback weeks.")
                else:
                    # Conviction filter
                    conv_filter = st.radio(
                        "Filter by conviction",
                        ["All", "🔥 HIGH", "⭐ MED", "👀 WATCH"],
                        horizontal=True,
                        key="conv_filter"
                    )

                    show_picks = picks_df.copy()
                    if conv_filter != "All":
                        level = conv_filter.split(" ")[-1]  # HIGH / MED / WATCH
                        show_picks = show_picks[show_picks["conviction"] == level]

                    if show_picks.empty:
                        st.info(f"No {conv_filter} conviction picks found.")
                    else:
                        # Conviction badge styling
                        def style_conviction(val):
                            colors = {"HIGH": "#16a34a", "MED": "#2563eb", "WATCH": "#d97706"}
                            c = colors.get(val, "#6b7280")
                            return f"background-color:{c};color:white;border-radius:4px;padding:2px 6px;font-weight:700"

                        # Format for display
                        disp = show_picks[[
                            "symbol","conviction","rs_now","rs_gap",
                            "rs_slope","cross_weeks_ago","close","sustained","sector"
                        ]].copy()

                        disp["cross_weeks_ago"] = disp["cross_weeks_ago"].apply(
                            lambda x: "This week" if x == 1 else f"{int(x)}w ago"
                        )
                        disp["rs_now"]   = disp["rs_now"].apply(lambda x: f"{x:+.4f}")
                        disp["rs_gap"]   = disp["rs_gap"].apply(lambda x: f"{x:+.4f}")
                        disp["rs_slope"] = disp["rs_slope"].apply(lambda x: f"{x:+.4f}")
                        disp["close"]    = disp["close"].apply(lambda x: f"₹{x:,.2f}")
                        disp["sustained"]= disp["sustained"].apply(lambda x: "✅" if x else "⏳")

                        disp.columns = [
                            "Symbol","Conviction","RS Now","Gap",
                            "Slope","Crossed","Close","Sustained","Sector"
                        ]
                        disp = disp.reset_index(drop=True)
                        disp.index = disp.index + 1  # rank from 1

                        st.dataframe(disp, use_container_width=True, height=400)

                        # Download top picks
                        st.download_button(
                            "⬇️ Download Top Picks CSV",
                            data      = picks_df.to_csv(index=False),
                            file_name = f"top_picks_{cross_run_id}.csv",
                            mime      = "text/csv",
                            key       = "dl_picks"
                        )

                    # ── Conviction breakdown ───────────────────────
                    st.markdown("---")
                    st.markdown("#### Conviction Breakdown")
                    ca, cb, cc = st.columns(3)
                    high_n  = len(picks_df[picks_df["conviction"] == "HIGH"])
                    med_n   = len(picks_df[picks_df["conviction"] == "MED"])
                    watch_n = len(picks_df[picks_df["conviction"] == "WATCH"])
                    ca.metric("🔥 HIGH",  high_n,  help="RS>0.5, Gap>0.1, Slope>0.1 — act first")
                    cb.metric("⭐ MED",   med_n,   help="RS>0.2, Gap>0.05 — monitor closely")
                    cc.metric("👀 WATCH", watch_n, help="Early stage — wait for confirmation")

                    # ── Sector chart of top picks ──────────────────
                    if not picks_df.empty:
                        st.markdown("---")
                        st.markdown("#### Sector Spread — Top Picks")
                        sec_counts = picks_df["sector"].value_counts().reset_index()
                        sec_counts.columns = ["Sector","Count"]
                        fig_sec = px.bar(
                            sec_counts, x="Count", y="Sector",
                            orientation="h",
                            color="Count",
                            color_continuous_scale="Teal",
                            title="Sectors in Top Picks",
                        )
                        fig_sec.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#0e1117",
                            plot_bgcolor="#0e1117",
                            showlegend=False,
                            coloraxis_showscale=False,
                            margin=dict(l=10,r=10,t=40,b=10),
                            height=350,
                        )
                        st.plotly_chart(fig_sec, use_container_width=True)

                st.markdown("---")

                # ══ ALL CROSSOVERS SECTION ═══════════════════════
                with st.expander(f"📋 All {len(cross_df)} Crossovers (full detail)", expanded=False):

                    # Filters
                    fa, fb, fc = st.columns(3)
                    with fa:
                        week_filter = st.multiselect(
                            "Crossed",
                            options=sorted(cross_df["cross_weeks_ago"].unique()),
                            default=sorted(cross_df["cross_weeks_ago"].unique()),
                            format_func=lambda x: "This week" if x == 1 else f"{int(x)} weeks ago",
                            key="week_filter"
                        )
                    with fb:
                        sust_filter = st.radio(
                            "Sustained",
                            ["All", "Sustained only", "Fresh only"],
                            key="sust_filter"
                        )
                    with fc:
                        rs_positive = st.checkbox("RS > 0 only", value=False, key="rs_pos_filter")

                    all_disp = cross_df.copy()
                    if week_filter:
                        all_disp = all_disp[all_disp["cross_weeks_ago"].isin(week_filter)]
                    if sust_filter == "Sustained only":
                        all_disp = all_disp[all_disp["sustained"] == True]
                    elif sust_filter == "Fresh only":
                        all_disp = all_disp[all_disp["sustained"] == False]
                    if rs_positive:
                        all_disp = all_disp[all_disp["rs_now"] > 0]

                    all_disp = all_disp.copy()
                    all_disp["cross_weeks_ago"] = all_disp["cross_weeks_ago"].apply(
                        lambda x: "This week" if x == 1 else f"{int(x)}w ago"
                    )
                    all_disp["rs_now"]    = all_disp["rs_now"].apply(lambda x: f"{x:+.4f}")
                    all_disp["rs_gap"]    = all_disp["rs_gap"].apply(lambda x: f"{x:+.4f}")
                    all_disp["rs_slope"]  = all_disp["rs_slope"].apply(lambda x: f"{x:+.4f}")
                    all_disp["close"]     = all_disp["close"].apply(lambda x: f"₹{x:,.2f}")
                    all_disp["sustained"] = all_disp["sustained"].apply(lambda x: "✅" if x else "⏳")
                    all_disp = all_disp[[
                        "symbol","rs_now","rs_ma_now","rs_gap","rs_slope",
                        "cross_weeks_ago","close","sustained","sector"
                    ]]
                    all_disp.columns = [
                        "Symbol","RS Now","RS MA","Gap","Slope",
                        "Crossed","Close","Sustained","Sector"
                    ]
                    st.markdown(f"**{len(all_disp)} stocks** after filters")
                    st.dataframe(all_disp, use_container_width=True, height=500)

                    st.download_button(
                        "⬇️ Download All Crossovers CSV",
                        data      = cross_df.to_csv(index=False),
                        file_name = f"crossover_{cross_run_id}.csv",
                        mime      = "text/csv",
                        key       = "dl_all_cross"
                    )

            else:
                cross_status.error("❌ Scan failed or returned no results. Check log above.")

        except Exception as e:
            cross_status.error(f"Error: {e}")
            st.exception(e)

    else:
        # Instruction state
        st.markdown("""
**How it works:**
1. Adjust **Lookback weeks** and **Min RS** in the sidebar
2. Click **🔀 Run Crossover Scanner**
3. Wait ~25 minutes (same as full screener)
4. Results show in two sections:

| Section | What you'll see |
|---|---|
| 🏆 Top Picks | Filtered, ranked, conviction-tagged setups |
| 📋 All Crossovers | Every stock with a recent RS/RSMA cross |
        """)

        st.markdown("---")
        st.markdown("#### What to look for in Top Picks")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
**🔥 HIGH conviction**
- RS Now > 0.5
- Gap above MA > 0.1
- Slope still rising

*Act on these first*
            """)
        with col2:
            st.markdown("""
**⭐ MED conviction**
- RS Now > 0.2
- Gap above MA > 0.05

*Monitor closely,
enter on pullback*
            """)
        with col3:
            st.markdown("""
**👀 WATCH**
- Early stage crossover
- Weak or flat slope

*Wait for another
week of confirmation*
            """)

        # Show most recent crossover file if exists
        cross_files = sorted(
            glob.glob(os.path.join(RESULTS_DIR, "crossover_*.csv")),
            reverse=True
        )
        if cross_files:
            latest_cross = cross_files[0]
            st.markdown("---")
            st.info(f"💡 Last scan: **{os.path.basename(latest_cross)}** — click Run to refresh")
