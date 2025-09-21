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
    
        # Загальний шлях
        total_distance = df['Distance(km)'].iloc[-1]
    
        # Загальний час
        #total_time = (df['Time(HH:mm:ss.fff)'].iloc[-1] - df['Time(HH:mm:ss.fff)'].iloc[0]).total_seconds() / 3600  # hours
    
        # Час у русі (беремо точки зі швидкістю > 1 км/год)
        #moving_time = (df.loc[df['Speed(km/h)'] > 1, 'time'].iloc[-1] - 
        #               df.loc[df['Speed(km/h)'] > 1, 'time'].iloc[0]).total_seconds() / 3600
    
        # Швидкість
        max_speed = df['Speed(km/h)'].max()
        avg_speed = df['Speed(km/h)'].mean()
    
        # Потужність
        max_power = df['MotorPower(W)'].max()
        avg_power = df['MotorPower(W)'].mean()
    
        # Ампераж
        max_current = df['Current(A)'].max()
        avg_current = df['Current(A)'].mean()
    
        # Асистент у %
        assist_percent = df['AssistLevel'].value_counts(normalize=True) * 100
    
        # Використаний заряд (по інтеграції)
        avg_voltage = df['Voltage(V)'].mean()
        #amp_hours = (df['Current(A)'].mean() * total_time)  # приблизно
        #watt_hours = (df['MotorPower(W)'].sum() / len(df)) * total_time
    
        # --- Output ---
        st.metric("Total Distance", f"{total_distance:.2f} km")
        #st.metric("Total Time", f"{total_time:.2f} h")
        #st.metric("Moving Time", f"{moving_time:.2f} h")
    
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Speed", f"{max_speed:.1f} km/h")
            st.metric("Avg Speed", f"{avg_speed:.1f} km/h")
        with col2:
            st.metric("Max Power", f"{max_power:.0f} W")
            st.metric("Avg Power", f"{avg_power:.0f} W")
        with col3:
            st.metric("Max Current", f"{max_current:.1f} A")
            st.metric("Avg Current", f"{avg_current:.1f} A")
    
        st.subheader("⚡ Assist Usage (%)")
        st.bar_chart(assist_percent)
    
        st.subheader("🔋 Energy Consumption")
        st.write(f"Average Voltage: {avg_voltage:.1f} V")
        st.write(f"Consumed Charge: {amp_hours:.2f} Ah")
        st.write(f"Consumed Energy: {watt_hours:.1f} Wh")
        st.write(f"Specific Consumption: {watt_hours / total_distance:.1f} Wh/km")
    
    # ============= DATA ==================
    with tabs[1]:
        st.subheader("📊 Data")
        st.dataframe(df)

        # coordinates
        lat_col, lon_col = "Latitude", "Longitude"
        df = df.dropna(subset=[lat_col, lon_col])
        st.write(f"Found {len(df)} records.")


    # ============= ROUTE ==================
    with tabs[2]:
        st.subheader("🗺️ Route on map")
        if not df.empty:
            df_clean = df #clean_gps(df) 
            start_coords = (df_clean[lat_col].iloc[0], df[lon_col].iloc[0])
            trip_map = folium.Map(location=start_coords, zoom_start=14)

            coords = df_clean[[lat_col, lon_col]].values.tolist()
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
