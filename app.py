import pandas as pd
import matplotlib.pyplot as plt
import folium
import streamlit as st
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Haversine distance in km
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def clean_gps(df, lat_col="Latitude", lon_col="Longitude", time_col="Time(HH:mm:ss.fff)", max_speed=50):
    df = df.copy().dropna(subset=[lat_col, lon_col])
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    mask = [True]  # first point is always ok
    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i-1][[lat_col, lon_col]]
        lat2, lon2 = df.iloc[i][[lat_col, lon_col]]
        t1, t2 = df.iloc[i-1][time_col], df.iloc[i][time_col]
        dt = (t2 - t1).total_seconds() / 3600.0  # hours

        if dt <= 0:
            mask.append(False)
            continue

        dist = haversine(lat1, lon1, lat2, lon2)  # km
        speed = dist / dt if dt > 0 else 0

        if speed > max_speed:
            print(f"Skipping point {i} (jump {dist:.2f} km in {dt*60:.1f} min at speed {speed:.1f} km/h)") 
            mask.append(False)  # skip "jump"
        else:
            mask.append(True)

    return df[mask].reset_index(drop=True)

st.set_page_config(layout="centered", page_icon="🚲", page_title="Eggrider Trip Analyzer")
st.title("🚲 Eggrider Trip Analyzer")

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
        "📊 Data",
        "🛣️ Route",
        "⚡ Speed & Power",
        "🔋 Voltage & Current",
        "📈 Assist Level",
    ])

    # ============= DATA ==================
    with tabs[0]:
        st.subheader("📊 Data")
        st.dataframe(df)

        # coordinates
        lat_col, lon_col = "Latitude", "Longitude"
        df = df.dropna(subset=[lat_col, lon_col])
        st.write(f"Found {len(df)} records.")


    # ============= ROUTE ==================
    with tabs[1]:
        st.subheader("🗺️ Route on map")
        if not df.empty:
            df_clean = df #clean_gps(df) 
            start_coords = (df_clean[lat_col].iloc[0], df[lon_col].iloc[0])
            trip_map = folium.Map(location=start_coords, zoom_start=14)

            coords = df_clean[[lat_col, lon_col]].values.tolist()
            folium.PolyLine(coords, color="blue", weight=3).add_to(trip_map)
            folium.Marker(coords[0], tooltip="Start").add_to(trip_map)
            folium.Marker(coords[-1], tooltip="End").add_to(trip_map)

            map_html = "trip_map.html"
            trip_map.save(map_html)
            st.components.v1.html(open(map_html, "r", encoding="utf-8").read(), height=600)

    # ============= SPEED + POWER ==================
    with tabs[2]:
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
    with tabs[3]:
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
    with tabs[4]:
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
