"""Streamlit dashboard: predict demand for a location/time, allocate drivers, inspect the model."""
import os
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as C
from allocation import allocate, slot_frame

st.set_page_config(page_title="Taxi Demand & Driver Allocation", page_icon="🚕", layout="wide")
DAYS = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}
NYC = {"lat": 40.73, "lon": -73.95}


@st.cache_data(show_spinner="Loading from HDFS...")
def load(name):
    uri = f"{C.RESULTS}/{name}"
    u = urlparse(uri)
    if u.scheme == "hdfs":
        import pyarrow.parquet as pq
        from pyarrow import fs
        return pq.read_table(u.path, filesystem=fs.HadoopFileSystem(u.hostname or "default", u.port or 0)).to_pandas()
    return pd.read_parquet(u.path if u.scheme == "file" else uri)


try:
    grid, zones, supply = load("pred_grid"), load("zones"), load("supply_proxy")
    metrics, test_preds, profile = load("metrics"), load("test_preds"), load("demand_profile")
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load results from `{C.RESULTS}`: {e}\n\nRun `run_all.sh` first and make sure HDFS is up.")
    st.stop()

# ---------- sidebar: global controls ----------
with st.sidebar:
    st.header("🚕 Controls")
    day = st.selectbox("Day of week", list(DAYS), index=4, format_func=DAYS.get)
    hour = st.slider("Hour of day", 0, 23, 18)
    fleet = st.number_input("Fleet size (drivers)", 100, 50000, C.DEFAULT_FLEET, step=100)
    tpd = st.slider("Trips per driver per hour", 0.5, 6.0, C.DEFAULT_TRIPS_PER_DRIVER, step=0.5)
    st.caption("Pipeline: CSV → HDFS → PySpark → GBT model → Streamlit")

slot = slot_frame(grid, zones, supply, day, hour)
st.title("Taxi Demand & Driver Allocation")
st.caption(f"{DAYS[day]} {hour:02d}:00 · ~1 km grid zones · typical-week prediction")
t_pred, t_alloc, t_model, t_an = st.tabs(["Predict demand", "Allocate drivers", "Model quality", "Demand analysis"])

# ---------- tab 1: predict demand for a location ----------
with t_pred:
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude", C.LAT_MIN, C.LAT_MAX, 40.758, format="%.4f")
    lon = c2.number_input("Longitude", C.LON_MIN, C.LON_MAX, -73.9855, format="%.4f")
    zid = C.zone_id(lon, lat)
    if zid not in set(zones.zone_id):
        near = zones.assign(d=(zones.lon - lon) ** 2 + (zones.lat - lat) ** 2).nsmallest(1, "d").iloc[0]
        zid = int(near.zone_id)
        st.warning(f"That point is outside the {len(zones)} modelled zones (low-demand area). Using nearest zone {zid}.")
    row = slot[slot.zone_id == zid].iloc[0]
    pct = (slot.prediction < row.prediction).mean() * 100

    k1, k2, k3 = st.columns(3)
    k1.metric("Predicted trips this hour", f"{row.prediction:,.0f}")
    k2.metric("Typical (historical avg)", f"{row.hist_mean:,.0f}")
    k3.metric("Busier than", f"{pct:.0f}% of zones")

    left, right = st.columns(2)
    day_curve = grid[(grid.zone_id == zid) & (grid.dow == day)].sort_values("hour")
    fig = go.Figure()
    fig.add_scatter(x=day_curve.hour, y=day_curve.prediction, name="Model", mode="lines+markers")
    fig.add_scatter(x=day_curve.hour, y=day_curve.hist_mean, name="Historical avg", line=dict(dash="dot"))
    fig.add_vline(x=hour, line_color="red", line_dash="dash")
    fig.update_layout(title=f"Zone {zid} on {DAYS[day]}", xaxis_title="Hour", yaxis_title="Trips", height=420)
    left.plotly_chart(fig, width="stretch")

    fig = px.scatter_map(slot, lat="lat", lon="lon", color="prediction", size=slot.prediction + 1,
                         color_continuous_scale="YlOrRd", zoom=9.5, center=NYC, height=420,
                         hover_name="zone_id", hover_data={"lat": False, "lon": False})
    fig.add_trace(go.Scattermap(lat=[lat], lon=[lon], mode="markers", marker=dict(size=16, color="blue"), name="Your point"))
    fig.update_layout(map_style="carto-positron", margin=dict(l=0, r=0, t=30, b=0), title="All zones this hour")
    right.plotly_chart(fig, width="stretch")

# ---------- tab 2: driver allocation ----------
with t_alloc:
    z, mv = allocate(slot, fleet, tpd)
    k = st.columns(5)
    k[0].metric("Predicted trips", f"{z.prediction.sum():,.0f}")
    k[1].metric("Drivers needed", f"{z.needed.sum():,}")
    k[2].metric("Fleet", f"{fleet:,}")
    k[3].metric("Drivers to move", f"{mv.drivers.sum():,}")
    k[4].metric("Unmet need after moves", f"{z.short_after.sum():,}")

    fig = px.scatter_map(z, lat="lat", lon="lon", color="gap", size=z.needed + 1, zoom=9.5, center=NYC, height=480,
                         color_continuous_scale="RdBu_r", color_continuous_midpoint=0, hover_name="zone_id",
                         hover_data={"needed": True, "current": True, "assigned": True, "lat": False, "lon": False})
    fig.update_layout(map_style="carto-positron", margin=dict(l=0, r=0, t=30, b=0),
                      title="Red = needs more drivers · Blue = surplus drivers")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Recommended moves")
    st.dataframe(mv.sort_values("drivers", ascending=False).head(50), width="stretch", hide_index=True)
    st.download_button("Download all moves (CSV)", mv.to_csv(index=False).encode(), "moves.csv", "text/csv")
    with st.expander("Per-zone table"):
        st.dataframe(z[["zone_id", "prediction", "needed", "current", "assigned", "gap", "short_now", "short_after"]]
                     .sort_values("needed", ascending=False), width="stretch", hide_index=True)
    st.caption("Supply is a proxy: drivers are assumed to be where trips ended in the previous hour. "
               "Trips-per-driver is an assumption, so treat results as a what-if tool.")

# ---------- tab 3: model quality ----------
with t_model:
    m = metrics.set_index("model")
    gain = 1 - m.mae["gbt_model"] / m.mae["baseline_zone_hour_dow_mean"]
    a, b = st.columns([2, 1])
    a.dataframe(metrics.round(3), width="stretch", hide_index=True)
    b.metric("MAE vs zone-hour-dow baseline", f"{gain:+.1%}")
    if gain <= 0:
        st.warning("The model does not beat the simple historical-average baseline.")
    st.caption("Time-based split: trained on Jan–May 2016, tested on June 2016. The data has no weather or events, "
               "so expect modest gains over the baseline.")

    tp = test_preds.assign(date=pd.to_datetime(test_preds.ts).dt.date)
    daily = tp.groupby("date")[["demand", "prediction"]].sum().reset_index()
    st.plotly_chart(px.line(daily, x="date", y=["demand", "prediction"], title="City-wide daily trips: actual vs predicted"),
                    width="stretch")
    top_zone = int(tp.groupby("zone_id").demand.sum().idxmax())
    zsel = st.selectbox("Zone", sorted(tp.zone_id.unique()), index=sorted(tp.zone_id.unique()).index(top_zone))
    zt = tp[tp.zone_id == zsel].sort_values("ts").tail(24 * 7)
    st.plotly_chart(px.line(zt, x="ts", y=["demand", "prediction"], title=f"Zone {zsel}: last 7 days of June, hourly"),
                    width="stretch")

# ---------- tab 4: analysis ----------
with t_an:
    pv = profile.pivot(index="dow", columns="hour", values="avg_city_trips").rename(index=DAYS)
    st.plotly_chart(px.imshow(pv, aspect="auto", color_continuous_scale="YlOrRd", height=380,
                              labels=dict(x="Hour", y="", color="Avg trips"), title="Average city-wide trips by day and hour"),
                    width="stretch")
    top = zones.nlargest(15, "total_trips").assign(zone=lambda d: "Zone " + d.zone_id.astype(str))
    st.plotly_chart(px.bar(top.sort_values("total_trips"), x="total_trips", y="zone", orientation="h",
                           title="Top 15 pickup zones (Jan–Jun 2016)"), width="stretch")
