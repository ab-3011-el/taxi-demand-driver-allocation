import glob
import hashlib
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Taxi Demand & Driver Allocation",
    page_icon="🚕",
    layout="wide",
)

BASE = "/rides"
RESULTS = f"{BASE}/results"

DAYS = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday",
}

NYC = {"lat": 40.73, "lon": -73.95}

# Friend pipeline uses a 0.01-degree grid.
LON_MIN, LON_MAX = -74.05, -73.70
LAT_MIN, LAT_MAX = 40.55, 40.95
CELL = 0.01

DEFAULT_FLEET = 3000
DEFAULT_TRIPS_PER_DRIVER = 2.0


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }

    .subtitle {
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 10px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# HDFS -> local Parquet loader
#
# We deliberately do NOT use pyarrow.fs.HadoopFileSystem.
# That caused the segmentation fault in WSL.
# ---------------------------------------------------------
@st.cache_data(show_spinner="Loading data from HDFS...")
def load_parquet_from_hdfs(dataset):
    source = f"{RESULTS}/{dataset}"

    cache_root = os.path.join(
        tempfile.gettempdir(),
        "taxi_dashboard_cache",
    )

    os.makedirs(cache_root, exist_ok=True)

    # Give each dataset a stable local cache directory.
    key = hashlib.md5(source.encode()).hexdigest()[:12]
    local_dir = os.path.join(cache_root, f"{dataset}_{key}")

    # Refresh the local copy if it doesn't exist.
    if not os.path.isdir(local_dir) or not glob.glob(
        os.path.join(local_dir, "*.parquet")
    ):
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)

        result = subprocess.run(
            ["hdfs", "dfs", "-get", source, local_dir],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Could not copy {source} from HDFS.\n"
                f"{result.stderr}"
            )

    files = sorted(glob.glob(os.path.join(local_dir, "*.parquet")))

    if not files:
        raise RuntimeError(f"No Parquet files found in {source}")

    frames = [pd.read_parquet(f) for f in files]

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------
# Load all friend-pipeline datasets
# ---------------------------------------------------------
@st.cache_data(show_spinner="Preparing dashboard data...")
def load_all():
    grid = load_parquet_from_hdfs("pred_grid")
    zones = load_parquet_from_hdfs("zones")
    supply = load_parquet_from_hdfs("supply_proxy")
    metrics = load_parquet_from_hdfs("metrics")
    test_preds = load_parquet_from_hdfs("test_preds")
    profile = load_parquet_from_hdfs("demand_profile")

    return grid, zones, supply, metrics, test_preds, profile


try:
    (
        grid,
        zones,
        supply,
        metrics,
        test_preds,
        profile,
    ) = load_all()

except Exception as e:
    st.error(
        "Could not load the new taxi pipeline results from HDFS."
    )
    st.code(str(e))
    st.info(
        "Make sure Hadoop is running and the friend pipeline "
        "has been executed."
    )
    st.stop()


# ---------------------------------------------------------
# Data preparation
# ---------------------------------------------------------
grid["zone_id"] = grid["zone_id"].astype(int)
grid["dow"] = grid["dow"].astype(int)
grid["hour"] = grid["hour"].astype(int)

zones["zone_id"] = zones["zone_id"].astype(int)

for df in [grid, zones, supply, metrics, test_preds, profile]:
    if "zone_id" in df.columns:
        df["zone_id"] = df["zone_id"].astype(int)


def zone_coordinates(zone_id):
    """
    Convert the pipeline's zone ID back to approximate
    zone-center latitude/longitude.
    """
    zone_x = int(zone_id) // 100
    zone_y = int(zone_id) % 100

    lon = LON_MIN + zone_x * CELL + CELL / 2
    lat = LAT_MIN + zone_y * CELL + CELL / 2

    return lat, lon


if "lat" not in zones.columns or "lon" not in zones.columns:
    coords = zones["zone_id"].apply(zone_coordinates)

    zones["lat"] = coords.apply(lambda x: x[0])
    zones["lon"] = coords.apply(lambda x: x[1])


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("🚕 Dashboard Controls")

    day = st.selectbox(
        "Day of week",
        options=list(DAYS.keys()),
        index=4,
        format_func=lambda x: DAYS[x],
    )

    hour = st.slider(
        "Hour of day",
        min_value=0,
        max_value=23,
        value=18,
    )

    fleet = st.number_input(
        "Fleet size (drivers)",
        min_value=100,
        max_value=50000,
        value=DEFAULT_FLEET,
        step=100,
    )

    trips_per_driver = st.slider(
        "Trips per driver per hour",
        min_value=0.5,
        max_value=6.0,
        value=DEFAULT_TRIPS_PER_DRIVER,
        step=0.5,
    )

    st.divider()

    st.caption(
        "Pipeline: CSV → HDFS → PySpark → GBT model → "
        "driver allocation → Streamlit"
    )

    if st.button("🔄 Clear dashboard cache"):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------
# Current slot
# ---------------------------------------------------------
slot = grid[
    (grid["dow"] == day)
    & (grid["hour"] == hour)
].copy()

slot = slot.merge(
    zones[["zone_id", "lat", "lon", "total_trips"]],
    on="zone_id",
    how="left",
)

# Supply proxy
slot = slot.merge(
    supply[
        (supply["dow"] == day)
        & (supply["hour"] == hour)
    ][["zone_id", "avg_dropoffs"]],
    on="zone_id",
    how="left",
)

slot["avg_dropoffs"] = slot["avg_dropoffs"].fillna(0)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🚕 Taxi Demand & Driver Allocation System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="subtitle">'
    f'{DAYS[day]} at {hour:02d}:00 · '
    f'200 modelled NYC zones · GBT demand prediction'
    f'</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# KPI row
# ---------------------------------------------------------
predicted_trips = slot["prediction"].sum()

peak_row = (
    grid.groupby("hour")["prediction"]
    .sum()
    .reset_index()
    .sort_values("prediction", ascending=False)
    .iloc[0]
)

busiest_zone = (
    slot.sort_values("prediction", ascending=False)
    .iloc[0]
)

estimated_drivers = int(
    np.ceil(
        predicted_trips / trips_per_driver
    )
)

available_drivers = int(fleet)

shortage = max(
    0,
    estimated_drivers - available_drivers,
)

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Predicted trips",
    f"{predicted_trips:,.0f}",
)

k2.metric(
    "Peak hour",
    f"{int(peak_row.hour):02d}:00",
)

k3.metric(
    "Busiest zone",
    f"Zone {int(busiest_zone.zone_id)}",
)

k4.metric(
    "Drivers required",
    f"{estimated_drivers:,}",
)

k5.metric(
    "Fleet shortage",
    f"{shortage:,}",
)


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📍 Demand Prediction",
        "🚗 Driver Allocation",
        "🤖 Model Quality",
        "📊 Demand Analysis",
    ]
)


# =========================================================
# TAB 1 — DEMAND PREDICTION
# =========================================================
with tab1:
    st.markdown(
        '<div class="section-title">'
        'Predict demand for a location'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    lat = c1.number_input(
        "Latitude",
        min_value=LAT_MIN,
        max_value=LAT_MAX,
        value=40.758,
        format="%.4f",
    )

    lon = c2.number_input(
        "Longitude",
        min_value=LON_MIN,
        max_value=LON_MAX,
        value=-73.9855,
        format="%.4f",
    )

    def calculate_zone_id(lon_value, lat_value):
        x = int(np.floor((lon_value - LON_MIN) / CELL))
        y = int(np.floor((lat_value - LAT_MIN) / CELL))
        return x * 100 + y

    zid = calculate_zone_id(lon, lat)

    zone_ids = set(zones["zone_id"])

    if zid not in zone_ids:
        distances = (
            (zones["lon"] - lon) ** 2
            + (zones["lat"] - lat) ** 2
        )

        nearest_index = distances.idxmin()
        nearest = zones.loc[nearest_index]

        zid = int(nearest.zone_id)

        st.warning(
            f"The selected location is outside the modelled "
            f"zones. Using nearest zone {zid}."
        )

    selected = slot[
        slot["zone_id"] == zid
    ]

    if selected.empty:
        st.error("No prediction is available for this zone/time.")
    else:
        row = selected.iloc[0]

        pct = (
            slot["prediction"] < row["prediction"]
        ).mean() * 100

        a, b, c = st.columns(3)

        a.metric(
            "Predicted trips this hour",
            f"{row['prediction']:,.1f}",
        )

        b.metric(
            "Historical average",
            f"{row['hist_mean']:,.1f}",
        )

        c.metric(
            "Busier than",
            f"{pct:.0f}% of modelled zones",
        )

        left, right = st.columns(2)

        # Hourly curve for selected zone/day
        day_curve = grid[
            (grid["zone_id"] == zid)
            & (grid["dow"] == day)
        ].sort_values("hour")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=day_curve["hour"],
                y=day_curve["prediction"],
                mode="lines+markers",
                name="GBT prediction",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=day_curve["hour"],
                y=day_curve["hist_mean"],
                mode="lines",
                name="Historical average",
                line=dict(dash="dot"),
            )
        )

        fig.add_vline(
            x=hour,
            line_dash="dash",
        )

        fig.update_layout(
            title=f"Zone {zid} demand profile — {DAYS[day]}",
            xaxis_title="Hour",
            yaxis_title="Predicted trips",
            height=420,
            margin=dict(l=20, r=20, t=60, b=20),
        )

        left.plotly_chart(
            fig,
            width="stretch",
        )

        # Demand map
        map_df = slot.copy()

        fig = px.scatter_map(
            map_df,
            lat="lat",
            lon="lon",
            color="prediction",
            size=map_df["prediction"] + 1,
            color_continuous_scale="YlOrRd",
            zoom=9.5,
            center=NYC,
            height=420,
            hover_name="zone_id",
            hover_data={
                "prediction": ":.1f",
                "hist_mean": ":.1f",
                "lat": False,
                "lon": False,
            },
        )

        fig.add_trace(
            go.Scattermap(
                lat=[lat],
                lon=[lon],
                mode="markers",
                marker=dict(
                    size=16,
                ),
                name="Selected location",
            )
        )

        fig.update_layout(
            map_style="carto-positron",
            margin=dict(l=0, r=0, t=40, b=0),
            title="Predicted demand by zone",
        )

        right.plotly_chart(
            fig,
            width="stretch",
        )


# =========================================================
# TAB 2 — DRIVER ALLOCATION
# =========================================================
with tab2:
    st.markdown(
        '<div class="section-title">'
        'Driver allocation and redistribution'
        '</div>',
        unsafe_allow_html=True,
    )

    # Current drivers are represented by the supply proxy.
    alloc = slot.copy()

    alloc["needed"] = np.ceil(
        alloc["prediction"] / trips_per_driver
    ).astype(int)

    # Convert previous-hour dropoffs into a current-location
    # supply proxy.
    total_supply_proxy = alloc["avg_dropoffs"].sum()

    if total_supply_proxy > 0:
        alloc["current"] = np.floor(
            alloc["avg_dropoffs"]
            / total_supply_proxy
            * fleet
        ).astype(int)
    else:
        alloc["current"] = 0

    alloc["gap"] = (
        alloc["current"] - alloc["needed"]
    )

    alloc["short_now"] = (
        alloc["needed"] - alloc["current"]
    ).clip(lower=0)

    # Allocate surplus drivers to shortage zones using
    # nearest-zone distance.
    surplus = alloc[
        alloc["gap"] > 0
    ][
        ["zone_id", "lat", "lon", "gap"]
    ].copy()

    shortage_zones = alloc[
        alloc["gap"] < 0
    ][
        ["zone_id", "lat", "lon", "gap"]
    ].copy()

    moves = []

    surplus_available = {
        int(r.zone_id): int(r.gap)
        for r in surplus.itertuples()
    }

    for target in shortage_zones.itertuples():
        remaining = int(-target.gap)

        if remaining <= 0:
            continue

        candidates = []

        for source in surplus.itertuples():
            available = surplus_available.get(
                int(source.zone_id),
                0,
            )

            if available <= 0:
                continue

            distance = (
                (source.lat - target.lat) ** 2
                + (source.lon - target.lon) ** 2
            ) ** 0.5

            candidates.append(
                (
                    distance,
                    int(source.zone_id),
                    available,
                )
            )

        candidates.sort()

        for distance, source_id, available in candidates:
            if remaining <= 0:
                break

            move = min(
                remaining,
                available,
            )

            if move > 0:
                moves.append(
                    {
                        "from_zone": source_id,
                        "to_zone": int(target.zone_id),
                        "drivers": int(move),
                        "distance_degrees": distance,
                    }
                )

                surplus_available[source_id] -= move
                remaining -= move

    moves_df = pd.DataFrame(
        moves,
        columns=[
            "from_zone",
            "to_zone",
            "drivers",
            "distance_degrees",
        ],
    )

    if moves_df.empty:
        moves_df = pd.DataFrame(
            columns=[
                "from_zone",
                "to_zone",
                "drivers",
                "distance_degrees",
            ]
        )

    alloc["assigned"] = alloc["current"]

    for move in moves_df.itertuples():
        alloc.loc[
            alloc["zone_id"] == move.from_zone,
            "assigned",
        ] -= move.drivers

        alloc.loc[
            alloc["zone_id"] == move.to_zone,
            "assigned",
        ] += move.drivers

    alloc["short_after"] = (
        alloc["needed"] - alloc["assigned"]
    ).clip(lower=0)

    total_moves = int(
        moves_df["drivers"].sum()
    )

    unmet = int(
        alloc["short_after"].sum()
    )

    a, b, c, d, e = st.columns(5)

    a.metric(
        "Predicted trips",
        f"{predicted_trips:,.0f}",
    )

    b.metric(
        "Drivers needed",
        f"{alloc['needed'].sum():,}",
    )

    c.metric(
        "Fleet",
        f"{fleet:,}",
    )

    d.metric(
        "Drivers moved",
        f"{total_moves:,}",
    )

    e.metric(
        "Unmet need",
        f"{unmet:,}",
    )

    map_alloc = alloc.copy()

    fig = px.scatter_map(
        map_alloc,
        lat="lat",
        lon="lon",
        color="gap",
        size=map_alloc["needed"] + 1,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        zoom=9.5,
        center=NYC,
        height=500,
        hover_name="zone_id",
        hover_data={
            "needed": True,
            "current": True,
            "assigned": True,
            "short_after": True,
            "lat": False,
            "lon": False,
        },
    )

    fig.update_layout(
        map_style="carto-positron",
        margin=dict(l=0, r=0, t=40, b=0),
        title=(
            "Driver balance by zone "
            "(negative gap = shortage)"
        ),
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.subheader("Recommended driver movements")

    if moves_df.empty:
        st.info(
            "No driver movements were required for this "
            "time slot."
        )
    else:
        moves_display = moves_df.copy()

        moves_display["distance_km"] = (
            moves_display["distance_degrees"]
            * 111
        )

        moves_display = moves_display[
            [
                "from_zone",
                "to_zone",
                "drivers",
                "distance_km",
            ]
        ]

        st.dataframe(
            moves_display.sort_values(
                "drivers",
                ascending=False,
            ).head(50),
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download driver movements CSV",
            moves_display.to_csv(
                index=False
            ).encode(),
            "driver_movements.csv",
            "text/csv",
        )

    with st.expander("Per-zone allocation table"):
        st.dataframe(
            alloc[
                [
                    "zone_id",
                    "prediction",
                    "needed",
                    "current",
                    "assigned",
                    "gap",
                    "short_now",
                    "short_after",
                ]
            ].sort_values(
                "needed",
                ascending=False,
            ),
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "Driver supply is a proxy based on historical dropoffs. "
        "Trips per driver per hour and fleet size are assumptions, "
        "so allocation results should be treated as a what-if analysis."
    )


# =========================================================
# TAB 3 — MODEL QUALITY
# =========================================================
with tab3:
    st.markdown(
        '<div class="section-title">'
        'GBT model performance'
        '</div>',
        unsafe_allow_html=True,
    )

    metrics_display = metrics.copy()

    st.dataframe(
        metrics_display.round(3),
        width="stretch",
        hide_index=True,
    )

    if "model" in metrics.columns:
        metric_lookup = metrics.set_index("model")

        if (
            "gbt_model" in metric_lookup.index
            and "baseline_zone_hour_dow_mean"
            in metric_lookup.index
        ):
            gbt_mae = float(
                metric_lookup.loc[
                    "gbt_model",
                    "mae",
                ]
            )

            baseline_mae = float(
                metric_lookup.loc[
                    "baseline_zone_hour_dow_mean",
                    "mae",
                ]
            )

            improvement = (
                1 - gbt_mae / baseline_mae
            )

            if improvement >= 0:
                st.success(
                    f"GBT MAE is {improvement:.1%} "
                    "lower than the historical baseline."
                )
            else:
                st.warning(
                    f"GBT MAE is {abs(improvement):.1%} "
                    "higher than the historical baseline."
                )

    st.caption(
        "The model uses a time-based split: training data "
        "before June 2016 and test data from June 2016. "
        "The dataset does not include weather or event variables."
    )

    test = test_preds.copy()

    if "ts" in test.columns:
        test["ts"] = pd.to_datetime(test["ts"])

        daily = (
            test.groupby(
                test["ts"].dt.date
            )[["demand", "prediction"]]
            .sum()
            .reset_index()
        )

        fig = px.line(
            daily,
            x="ts",
            y=["demand", "prediction"],
            title="City-wide daily demand: actual vs predicted",
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Trips",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    if "zone_id" in test.columns:
        zone_totals = (
            test.groupby("zone_id")["demand"]
            .sum()
            .sort_values(ascending=False)
        )

        if len(zone_totals) > 0:
            default_zone = int(
                zone_totals.index[0]
            )

            selected_zone = st.selectbox(
                "Inspect test predictions for zone",
                sorted(
                    test["zone_id"].unique()
                ),
                index=sorted(
                    test["zone_id"].unique()
                ).index(default_zone),
            )

            zone_test = test[
                test["zone_id"] == selected_zone
            ].copy()

            if "ts" in zone_test.columns:
                zone_test = zone_test.sort_values(
                    "ts"
                ).tail(24 * 7)

                fig = px.line(
                    zone_test,
                    x="ts",
                    y=[
                        "demand",
                        "prediction",
                    ],
                    title=(
                        f"Zone {selected_zone}: "
                        "actual vs predicted"
                    ),
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )


# =========================================================
# TAB 4 — DEMAND ANALYSIS
# =========================================================
with tab4:
    st.markdown(
        '<div class="section-title">'
        'Demand patterns'
        '</div>',
        unsafe_allow_html=True,
    )

    profile_plot = profile.copy()

    profile_plot["day"] = (
        profile_plot["dow"]
        .map(DAYS)
    )

    pivot = profile_plot.pivot(
        index="day",
        columns="hour",
        values="avg_city_trips",
    )

    # Put days in natural Sunday-Saturday order.
    day_order = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ]

    pivot = pivot.reindex(
        [
            d
            for d in day_order
            if d in pivot.index
        ]
    )

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        height=400,
        labels={
            "x": "Hour",
            "y": "Day",
            "color": "Average trips",
        },
        title="Average city-wide trips by day and hour",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    # Hourly demand
    hourly = (
        grid.groupby("hour")["prediction"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        hourly,
        x="hour",
        y="prediction",
        markers=True,
        title="Predicted city-wide demand by hour",
    )

    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Predicted trips",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    # Top zones
    if "total_trips" in zones.columns:
        top = (
            zones.nlargest(
                15,
                "total_trips",
            )
            .copy()
        )

        top["zone"] = (
            "Zone "
            + top["zone_id"].astype(str)
        )

        fig = px.bar(
            top.sort_values(
                "total_trips"
            ),
            x="total_trips",
            y="zone",
            orientation="h",
            title="Top 15 pickup zones",
        )

        fig.update_layout(
            xaxis_title="Historical trips",
            yaxis_title="Zone",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "Taxi Demand & Driver Allocation System · "
    "PySpark + HDFS + GBT + Streamlit"
)
