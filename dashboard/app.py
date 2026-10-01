import io
import subprocess

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Taxi Demand & Driver Allocation",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DESIGN TOKENS
# ============================================================

INK = "#0f172a"
SLATE = "#64748b"
LINE = "#e2e8f0"
SURFACE = "#f8fafc"
TAXI = "#f5b800"
TEAL = "#0e7490"
RED = "#dc2626"
GREEN = "#16a34a"
FONT = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{ font-family: {FONT}; }}
.stApp {{ background-color: {SURFACE}; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px; }}
#MainMenu, footer, header {{ visibility: hidden; }}

section[data-testid="stSidebar"] {{ background-color: {INK}; }}
section[data-testid="stSidebar"] * {{ color: #e2e8f0; }}
section[data-testid="stSidebar"] hr {{ border-color: #1e293b; }}
.sidebar-brand {{ font-size: 1.25rem; font-weight: 700; color: #ffffff; }}
.sidebar-sub {{ font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.8rem; }}
.pipeline {{ font-size: 0.8rem; line-height: 1.9; color: #cbd5e1; }}
.pipeline b {{ color: {TAXI}; }}

.hero {{
    background: {INK}; border-left: 8px solid {TAXI}; border-radius: 10px;
    padding: 1.3rem 1.8rem; margin-bottom: 1rem;
}}
.hero-title {{ color: #fff; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; }}
.hero-sub {{ color: #94a3b8; font-size: 0.95rem; margin-top: 0.3rem; }}

.kpi {{
    background: #fff; border: 1px solid {LINE}; border-top: 4px solid {TAXI};
    border-radius: 10px; padding: 0.9rem 1.1rem; height: 100%;
}}
.kpi.alert {{ border-top-color: {RED}; }}
.kpi.ok {{ border-top-color: {GREEN}; }}
.kpi.teal {{ border-top-color: {TEAL}; }}
.kpi-label {{ color: {SLATE}; font-size: 0.82rem; font-weight: 500; }}
.kpi-value {{ color: {INK}; font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em; margin-top: 0.1rem; }}
.kpi-note {{ color: {SLATE}; font-size: 0.76rem; }}

.section-title {{ color: {INK}; font-size: 1.1rem; font-weight: 650; margin: 1.4rem 0 0.1rem 0; }}
.section-desc {{ color: {SLATE}; font-size: 0.88rem; margin-bottom: 0.6rem; }}

.insight {{
    background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px;
    padding: 0.8rem 1.1rem; color: #78350f; font-size: 0.92rem; margin: 0.8rem 0;
}}

div[data-testid="stPlotlyChart"] {{
    background: #fff; border: 1px solid {LINE}; border-radius: 10px; padding: 0.4rem;
}}
div[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 10px; overflow: hidden; }}
.footer {{ color: {SLATE}; font-size: 0.8rem; text-align: center; padding-top: 0.5rem; }}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================


def kpi(label, value, note="", kind=""):
    note_html = f'<div class="kpi-note">{note}</div>' if note else ""
    st.markdown(
        f'<div class="kpi {kind}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def section(title, description=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div class="section-desc">{description}</div>', unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)


def style_fig(fig, height=380, legend=True):
    fig.update_layout(
        height=height,
        font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
        title=dict(font=dict(size=15, color=INK), x=0.01),
        showlegend=legend,
        legend=dict(orientation="h", y=-0.2, title_text=""),
        hoverlabel=dict(font_family=FONT),
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="#f1f5f9", zeroline=False)
    return fig


def build_map_df(zones_df):
    df = zones_df.copy()
    df["zone_x"] = df["zone_id"] // 100
    df["zone_y"] = df["zone_id"] % 100
    df["longitude"] = -74.05 + df["zone_x"] * 0.05 + 0.025
    df["latitude"] = 40.55 + df["zone_y"] * 0.05 + 0.025
    df["map_size"] = df["total_trips"] / df["total_trips"].max() * 35 + 8
    return df


def zone_map(map_df, height=520):
    hover = {"latitude": False, "longitude": False, "map_size": False, "total_trips": ":,"}
    for col, fmt in [("avg_trip_duration_minutes", ":.2f"), ("avg_passengers", ":.2f")]:
        if col in map_df.columns:
            hover[col] = fmt
    fig = px.scatter_map(
        map_df, lat="latitude", lon="longitude", size="map_size", color="total_trips",
        hover_name="zone_id", hover_data=hover, zoom=9,
        center={"lat": 40.72, "lon": -73.98}, height=height,
        color_continuous_scale="YlOrRd",
        labels={"total_trips": "Trips", "avg_trip_duration_minutes": "Avg duration (min)",
                "avg_passengers": "Avg passengers"},
    )
    fig.update_layout(
        map_style="carto-positron",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="Trips", thickness=14),
    )
    return fig


def csv_download(df, label, filename, key):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv", key=key)


# ============================================================
# HDFS READER
# ============================================================


def read_hdfs_csv(path):
    try:
        result = subprocess.run(
            ["hdfs", "dfs", "-cat", f"{path}/part-*.csv"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
        )
        return pd.read_csv(io.StringIO(result.stdout))
    except Exception as e:
        st.error(f"Could not read `{path}` from HDFS: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Loading data from HDFS...")
def load_all():
    base = "/rides/results"
    return {
        "hourly": read_hdfs_csv(f"{base}/hourly_demand"),
        "daily": read_hdfs_csv(f"{base}/daily_demand"),
        "zones": read_hdfs_csv(f"{base}/zone_demand"),
        "predictions": read_hdfs_csv(f"{base}/demand_predictions"),
        "allocation": read_hdfs_csv(f"{base}/driver_allocation"),
    }


data = load_all()
hourly, daily, zones = data["hourly"], data["daily"], data["zones"]
predictions, allocation = data["predictions"], data["allocation"]

if hourly.empty or zones.empty:
    st.error("Dashboard data could not be loaded from HDFS. "
             "Make sure Hadoop is running and the Spark jobs have completed.")
    st.stop()

# ============================================================
# DATA PREPARATION
# ============================================================

hourly["pickup_hour"] = hourly["pickup_hour"].astype(int)
hourly = hourly.sort_values("pickup_hour")
zones["zone_id"] = zones["zone_id"].astype(int)
zones = zones.sort_values("total_trips", ascending=False).reset_index(drop=True)

if not daily.empty:
    daily["day_number"] = daily["day_number"].astype(int)
    daily = daily.sort_values("day_number")

if not allocation.empty:
    allocation["pickup_hour"] = allocation["pickup_hour"].astype(int)
    allocation = allocation.sort_values("pickup_hour")

if not predictions.empty:
    predictions["pickup_hour"] = predictions["pickup_hour"].astype(int)
    predictions["error"] = predictions["prediction"] - predictions["total_trips"]

# Demand level of each zone (based on all zones)
q1, q2 = zones["total_trips"].quantile([0.33, 0.66])
zones["demand_level"] = pd.cut(
    zones["total_trips"], bins=[-float("inf"), q1, q2, float("inf")],
    labels=["Low", "Medium", "High"],
).astype(str)

# ============================================================
# SIDEBAR: FILTERS ONLY (no page navigation)
# ============================================================

DEFAULTS = {"hour_range": (0, 23), "levels": ["High", "Medium", "Low"], "top_n": 10}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset_filters():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


with st.sidebar:
    st.markdown('<div class="sidebar-brand">🚕 Taxi Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Filters update every chart on the page</div>',
                unsafe_allow_html=True)

    hour_range = st.slider("Hour of day", 0, 23, key="hour_range",
                           help="Applies to demand, driver allocation and predictions.")
    levels = st.multiselect("Zone demand level", ["High", "Medium", "Low"], key="levels",
                            help="Applies to the zone map and zone ranking.")
    top_n = st.slider("Zones in ranking", 5, min(20, len(zones)), key="top_n")
    st.button("Reset filters", on_click=reset_filters, use_container_width=True)

    st.markdown("---")
    st.markdown(
        '<div class="pipeline"><b>Data pipeline</b><br>Taxi CSV → HDFS → Spark<br>→ PySpark ML → Streamlit</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# APPLY FILTERS
# ============================================================

h_lo, h_hi = hour_range
in_range = lambda df: df[(df["pickup_hour"] >= h_lo) & (df["pickup_hour"] <= h_hi)]

f_hourly = in_range(hourly)
f_alloc = in_range(allocation) if not allocation.empty else allocation
f_pred = in_range(predictions) if not predictions.empty else predictions
f_zones = zones[zones["demand_level"].isin(levels)] if levels else zones.iloc[0:0]

if f_hourly.empty:
    st.warning("No data for the selected hours.")
    st.stop()

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero"><div class="hero-title">Taxi Demand & Driver Allocation System</div>'
    '<div class="hero-sub">Big data analytics dashboard built on the NYC Taxi Trip dataset · '
    f'showing {h_lo:02d}:00 to {h_hi:02d}:59</div></div>',
    unsafe_allow_html=True,
)

# ============================================================
# KPI ROW
# ============================================================

total_all = int(hourly["total_trips"].sum())
total_sel = int(f_hourly["total_trips"].sum())
peak_row = f_hourly.loc[f_hourly["total_trips"].idxmax()]
peak_hour = int(peak_row["pickup_hour"])
peak_trips = int(peak_row["total_trips"])

if not f_zones.empty:
    top_zone = f_zones.iloc[0]
    zone_text, zone_note = f"Zone {int(top_zone['zone_id'])}", f"{int(top_zone['total_trips']):,} trips"
else:
    zone_text, zone_note = "-", "no zones selected"

if not f_alloc.empty:
    shortage = int((f_alloc["allocation_status"] == "Driver Shortage").sum())
    short_text, short_note = f"{shortage:,}", f"of {len(f_alloc):,} records"
else:
    shortage, short_text, short_note = 0, "-", "no data"

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    kpi("Trips in selection", f"{total_sel:,}", f"{total_sel / total_all * 100:.1f}% of all trips")
with k2:
    kpi("Peak hour", f"{peak_hour:02d}:00", f"{peak_trips:,} trips", "teal")
with k3:
    kpi("Busiest zone", zone_text, zone_note, "teal")
with k4:
    kpi("Avg trips per hour", f"{f_hourly['total_trips'].mean():,.0f}", f"{len(f_hourly)} hours selected")
with k5:
    kpi("Driver shortage cases", short_text, short_note, "alert")

# ============================================================
# ROW 1: DEMAND PATTERNS
# ============================================================

section("Demand patterns", "How taxi demand changes across the day and the week.")
c1, c2 = st.columns(2)

with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly["pickup_hour"], y=hourly["total_trips"], mode="lines",
        line=dict(color="#cbd5e1", width=2), name="All hours",
    ))
    fig.add_trace(go.Scatter(
        x=f_hourly["pickup_hour"], y=f_hourly["total_trips"], mode="lines+markers",
        line=dict(color=INK, width=3), marker=dict(color=INK, size=6),
        fill="tozeroy", fillcolor="rgba(245,184,0,0.35)", name="Selected hours",
    ))
    fig.add_vline(x=peak_hour, line_dash="dash", line_color=RED,
                  annotation_text=f"Peak {peak_hour:02d}:00", annotation_position="top")
    fig.update_layout(title="Hourly taxi demand", xaxis_title="Hour of day", yaxis_title="Trips",
                      xaxis=dict(tickmode="linear", dtick=2))
    st.plotly_chart(style_fig(fig), use_container_width=True)

with c2:
    if daily.empty:
        st.info("Daily demand data is not available.")
    else:
        peak_day = daily.loc[daily["total_trips"].idxmax(), "day_name"]
        colors = [TAXI if d == peak_day else "#cbd5e1" for d in daily["day_name"]]
        fig = go.Figure(go.Bar(x=daily["day_name"], y=daily["total_trips"], marker_color=colors))
        fig.update_layout(title=f"Daily taxi demand (busiest: {peak_day})",
                          xaxis_title="Day", yaxis_title="Trips")
        st.plotly_chart(style_fig(fig, legend=False), use_container_width=True)

insight(f"Within the selected hours demand peaks at <b>{peak_hour:02d}:00</b> with {peak_trips:,} trips. "
        f"The selection covers <b>{total_sel / total_all * 100:.1f}%</b> of all trips.")

# ============================================================
# ROW 2: GEOGRAPHY
# ============================================================

section("Geographic demand", "Use the zone demand level and ranking filters in the sidebar.")
m1, m2 = st.columns([3, 2])

with m1:
    if f_zones.empty:
        st.info("Select at least one zone demand level.")
    else:
        st.plotly_chart(zone_map(build_map_df(f_zones)), use_container_width=True)
        st.caption("Zones are project-defined grid cells created from pickup longitude and latitude.")

with m2:
    top = f_zones.head(top_n).copy()
    if top.empty:
        st.info("No zones to rank.")
    else:
        top["zone_label"] = "Zone " + top["zone_id"].astype(str)
        fig = px.bar(top.sort_values("total_trips"), x="total_trips", y="zone_label",
                     orientation="h", title=f"Top {len(top)} zones",
                     color="total_trips", color_continuous_scale="YlOrRd")
        fig.update_layout(xaxis_title="Total trips", yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, height=520, legend=False), use_container_width=True)

# ============================================================
# ROW 3: DRIVER ALLOCATION
# ============================================================

section("Driver allocation", "Drivers required to serve demand compared with drivers available.")

if f_alloc.empty:
    st.warning("Driver allocation data is not available for the selected hours.")
else:
    df = f_alloc.copy()
    df["time"] = df["pickup_hour"].astype(str).str.zfill(2) + ":00"
    sufficient = int((df["allocation_status"] == "Sufficient Drivers").sum())

    a1, a2 = st.columns([3, 1])
    with a1:
        long_df = df.melt(id_vars=["time"], value_vars=["required_drivers", "available_drivers"],
                          var_name="Series", value_name="Drivers")
        long_df["Series"] = long_df["Series"].map(
            {"required_drivers": "Required", "available_drivers": "Available"})
        fig = px.bar(long_df, x="time", y="Drivers", color="Series", barmode="group",
                     title="Required vs available drivers",
                     color_discrete_map={"Required": TAXI, "Available": TEAL})
        fig.update_layout(xaxis_title="Hour", yaxis_title="Drivers")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with a2:
        status = df["allocation_status"].value_counts().reset_index()
        status.columns = ["Status", "Count"]
        fig = px.pie(status, names="Status", values="Count", hole=0.55, title="Allocation status",
                     color="Status",
                     color_discrete_map={"Driver Shortage": RED, "Sufficient Drivers": GREEN})
        st.plotly_chart(style_fig(fig), use_container_width=True)

    if shortage:
        worst = df.assign(gap=df["required_drivers"] - df["available_drivers"]).nlargest(1, "gap").iloc[0]
        insight(f"Largest shortfall is at <b>{worst['time']}</b>: {int(worst['required_drivers']):,} drivers "
                f"required against {int(worst['available_drivers']):,} available.")
    st.caption("Project assumptions: 3 taxi trips per driver per hour and 3,000 available drivers.")

# ============================================================
# ROW 4: ML PREDICTION
# ============================================================

section("Demand prediction", "Random Forest model trained on pickup hour and day of week.")

if f_pred.empty:
    st.warning("Prediction data is not available for the selected hours.")
else:
    mae = f_pred["error"].abs().mean()
    mape = (f_pred["error"].abs() / f_pred["total_trips"].replace(0, pd.NA)).dropna().mean() * 100
    ss_res = (f_pred["error"] ** 2).sum()
    ss_tot = ((f_pred["total_trips"] - f_pred["total_trips"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

    p1, p2, p3 = st.columns(3)
    with p1:
        kpi("Mean absolute error", f"{mae:,.0f}", "trips per record", "teal")
    with p2:
        kpi("Mean % error", f"{mape:.1f}%", "lower is better", "teal")
    with p3:
        kpi("R² score", f"{r2:.3f}", "closer to 1 is better", "ok")

    lo = float(min(f_pred["total_trips"].min(), f_pred["prediction"].min()))
    hi = float(max(f_pred["total_trips"].max(), f_pred["prediction"].max()))

    st.write("")
    g1, g2 = st.columns(2)
    with g1:
        fig = px.scatter(f_pred, x="total_trips", y="prediction", color="pickup_hour",
                         title="Actual vs predicted demand", color_continuous_scale="YlOrRd",
                         labels={"total_trips": "Actual trips", "prediction": "Predicted trips",
                                 "pickup_hour": "Hour"})
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                                 line=dict(color=INK, dash="dash")))
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with g2:
        fig = px.histogram(f_pred, x="error", nbins=30, title="Prediction error distribution",
                           color_discrete_sequence=[TEAL])
        fig.update_layout(xaxis_title="Predicted minus actual trips", yaxis_title="Count")
        st.plotly_chart(style_fig(fig, legend=False), use_container_width=True)

# ============================================================
# DATA TABLES (collapsed)
# ============================================================

section("Underlying data")
with st.expander("Hourly demand"):
    st.dataframe(f_hourly, use_container_width=True, hide_index=True)
    csv_download(f_hourly, "Download CSV", "hourly_demand.csv", "dl_hourly")
with st.expander("Zone demand"):
    st.dataframe(f_zones, use_container_width=True, hide_index=True)
    csv_download(f_zones, "Download CSV", "zone_demand.csv", "dl_zones")
if not f_alloc.empty:
    with st.expander("Driver allocation"):
        st.dataframe(f_alloc, use_container_width=True, hide_index=True)
        csv_download(f_alloc, "Download CSV", "driver_allocation.csv", "dl_alloc")
if not f_pred.empty:
    with st.expander("Predictions"):
        st.dataframe(f_pred, use_container_width=True, hide_index=True)
        csv_download(f_pred, "Download CSV", "demand_predictions.csv", "dl_pred")

st.markdown("---")
st.markdown(
    '<div class="footer">Taxi Demand & Driver Allocation System · '
    "Hadoop · HDFS · Apache Spark · PySpark ML · Streamlit</div>",
    unsafe_allow_html=True,
)
