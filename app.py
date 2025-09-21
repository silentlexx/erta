import pandas as pd
import matplotlib.pyplot as plt
import folium
import streamlit as st
import numpy as np
from streamlit.components.v1 import html

st.set_page_config(layout="centered", page_icon="🚲", page_title="Eggrider Trip Analyzer")
st.title("🚲 Eggrider Trip Analyzer")

def fullscreen_html(html_code):
    return """
            <style>
            .st-btn {
                background-color: #fff;
                color: #000;
                border: 2px solid rgba(0, 0, 0, 0.2);
                border-radius: 2px;
                text-align: center;
                font-size: 22px;
                cursor: pointer;
                width: 40px;
                height: 40px;
                font-weight: bold;
                line-height: 28px;
                position: absolute;
                z-index: 9999;
                right: 15px;
                top: 15px;
            }
            .st-btn:hover {
                background-color: #f0f2f6;
                border-color: #c0c0c0;
            }
            </style>
            <button class="st-btn" onclick="openFullscreen()">&#x26F6;</button>
            <div id="map-container" style="width:100%; height:400px; background:lightblue; text-align:center; line-height:400px;">
                """ + html_code + """
            </div>
            <script>
            function openFullscreen() {
            var elem = document.getElementById("map-container");
            if (elem.requestFullscreen) {
                elem.requestFullscreen();
            } else if (elem.mozRequestFullScreen) { /* Firefox */
                elem.mozRequestFullScreen();
            } else if (elem.webkitRequestFullscreen) { /* Chrome, Safari & Opera */
                elem.webkitRequestFullscreen();
            } else if (elem.msRequestFullscreen) { /* IE/Edge */
                elem.msRequestFullscreen();
            }
            }
            </script>
            """

# ---------- helper to parse time ----------
def parse_time_column(df, time_col="Time(HH:mm:ss.fff)"):
    times = pd.to_timedelta(df[time_col])
    secs = times.dt.total_seconds().to_numpy()

    # handle day rollovers (00:00:00)
    day_offset = 0
    offsets = np.zeros(len(secs))
    prev = secs[0]
    for i in range(1, len(secs)):
        if secs[i] < prev:  # next day
            day_offset += 86400
        offsets[i] = day_offset
        prev = secs[i]

    base_date = pd.Timestamp("2000-01-01")
    ts = base_date + pd.to_timedelta(secs + offsets, unit="s")

    df = df.copy()
    df["timestamp"] = ts
    return df


# ---------- Statistics tab ----------
def render_statistics(df):
    # ensure timestamp
    if "timestamp" not in df.columns:
        df = parse_time_column(df, "Time(HH:mm:ss.fff)")

    # compute dt
    df["dt_s"] = df["timestamp"].diff().dt.total_seconds().clip(lower=0).fillna(0)
    df["dt_h"] = df["dt_s"] / 3600

    total_distance = df["Distance(km)"].iloc[-1]

    # total / moving time
    total_time = df["dt_h"].sum()
    if "Speed(km/h)" in df.columns:
        moving_time = df.loc[df["Speed(km/h)"] > 1, "dt_h"].sum()
    else:
        moving_time = total_time

    # speed
    max_speed = df["Speed(km/h)"].max() if "Speed(km/h)" in df.columns else None
    avg_speed = df["Speed(km/h)"].mean() if "Speed(km/h)" in df.columns else None

    # motor power
    max_power = df["MotorPower(W)"].max() if "MotorPower(W)" in df.columns else None
    avg_power = df["MotorPower(W)"].mean() if "MotorPower(W)" in df.columns else None

    # current
    max_current = df["Current(A)"].max() if "Current(A)" in df.columns else None
    avg_current = df["Current(A)"].mean() if "Current(A)" in df.columns else None

    # assist distribution
    assist_percent = None
    if "AssistLevel" in df.columns:
        assist_percent = df["AssistLevel"].value_counts(normalize=True).sort_index() * 100

    # energy consumption
    total_ah, total_wh, avg_voltage = None, None, None
    if "Current(A)" in df.columns:
        total_ah = (df["Current(A)"] * df["dt_h"]).sum()
    if "MotorPower(W)" in df.columns:
        total_wh = (df["MotorPower(W)"] * df["dt_h"]).sum()
    if "Voltage(V)" in df.columns:
        avg_voltage = df["Voltage(V)"].mean()

    wh_per_km = (total_wh / total_distance) if total_wh and total_distance > 0 else None

    # ---------- UI ----------

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Distance", f"{total_distance:.2f} km")
        st.metric("Total Time", f"{total_time:.2f} h")
        st.metric("Moving Time", f"{moving_time:.2f} h")
    with col2:
        if max_speed is not None:
            st.metric("Max Speed", f"{max_speed:.1f} km/h")
            st.metric("Avg Speed", f"{avg_speed:.1f} km/h")
        if max_power is not None:
            st.metric("Max Motor Power", f"{max_power:.0f} W")
            st.metric("Avg Motor Power", f"{avg_power:.0f} W")
    with col3:
        if max_current is not None:
            st.metric("Max Current", f"{max_current:.1f} A")
            st.metric("Avg Current", f"{avg_current:.1f} A")
        if avg_voltage is not None:
            st.metric("Avg Voltage", f"{avg_voltage:.1f} V")

    if assist_percent is not None:
        st.subheader("⚡ Assist Usage (%)")
        st.bar_chart(assist_percent)

    st.subheader("🔋 Energy Consumption")
    if total_ah is not None:
        st.write(f"Consumed Charge: **{total_ah:.2f} Ah**")
    if total_wh is not None:
        st.write(f"Consumed Energy: **{total_wh:.1f} Wh**")
    if wh_per_km is not None:
        st.write(f"Specific Consumption: **{wh_per_km:.1f} Wh/km**")
# ----------------------------
# UI
# ----------------------------

uploaded_file = st.file_uploader("Upload CSV from Eggrider", type=["csv"])

if uploaded_file:
    # read CSV
    df = pd.read_csv(uploaded_file, sep=";", skiprows=1)
    min_dist = float(df["Distance(km)"].min())
    max_dist = float(df["Distance(km)"].max())

    dist_range = st.slider(
        "Select distance range (km)",
        min_value=min_dist,
        max_value=max_dist,
        value=(min_dist, max_dist),  # initially full route
        step=0.1
    )

    df = df[
        (df["Distance(km)"] >= dist_range[0]) &
        (df["Distance(km)"] <= dist_range[1])
    ]

    # create tabs
    tabs = st.tabs([
        "📊 Statistics",
        "📊 Data",
        "🛣️ Route",
        "⚡ Speed & Power",
        "🔋 Voltage & Current",
        "📈 Assist Level",
    ])

    # ============= STAT ==================

    with tabs[0]:
        st.subheader("📊 Ride Statistics")
    
        render_statistics(df)

    # ============= ROUTE ==================
    with tabs[2]:
        st.subheader("🗺️ Route on map")
        if not df.empty:
            start_coords = (df["Latitude"].iloc[0], df["Longitude"].iloc[0])
            trip_map = folium.Map(location=start_coords, zoom_start=14)

            coords = df[["Latitude", "Longitude"]].values.tolist()
            folium.PolyLine(coords, color="blue", weight=3).add_to(trip_map)
            folium.Marker(coords[0], tooltip="Start").add_to(trip_map)
            folium.Marker(coords[-1], tooltip="End").add_to(trip_map)

            html(fullscreen_html(trip_map._repr_html_()), height=600)

    # ============= SPEED + POWER ==================
    with tabs[3]:
        st.subheader("⚡ Speed & Power")
        fig, ax1 = plt.subplots()

        ax1.plot(df["Distance(km)"], df["Speed(km/h)"], label="Speed (km/h)", color="blue")
        ax1.plot(df["Distance(km)"], df["SpeedGPS(km/h)"], label="GPS Speed (km/h)", color="green", alpha=0.6)
        ax1.set_ylabel("Speed (km/h)", color="blue")
        ax1.legend()


        ax2 = ax1.twinx()
        ax2.plot(df["Distance(km)"], df["MotorPower(W)"], label="Motor Power (W)", color="red", alpha=0.6)
        ax2.set_ylabel("Power (W)", color="red")

        ax1.set_xlabel("Distance (km)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ============= VOLTAGE / CURRENT ==================
    with tabs[4]:
        st.subheader("🔋 Voltage & Current")
        fig, ax1 = plt.subplots()

        ax1.plot(df["Distance(km)"], df["Voltage(V)"], color="orange", label="Voltage (V)")
        ax1.set_ylabel("Voltage (V)", color="orange")
        ax1.set_ylim(41, 54)

        ax2 = ax1.twinx()
        ax2.plot(df["Distance(km)"], df["Current(A)"], color="red", label="Current (A)", alpha=0.6)
        ax2.set_ylabel("Current (A)", color="red")

        ax1.set_xlabel("Distance (km)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ============= ASSIST LEVEL ==================
    with tabs[5]:
        st.subheader("📈 Assist Level")
        fig, ax1 = plt.subplots()

        ax1.plot(df["Distance(km)"], df["AssistLevel"], label="PAS Level", color="brown")
        ax1.set_ylabel("Assist Level", color="brown")
        ax1.set_ylim(0, 9)


        ax2 = ax1.twinx()
        ax2.plot(df["Distance(km)"], df["MotorPower(W)"], label="Motor Power (W)", color="red", alpha=0.6)
        ax2.set_ylabel("Power (W)", color="red")

        ax1.set_xlabel("Distance (km)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
