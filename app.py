import pandas as pd
import matplotlib.pyplot as plt
import folium
import streamlit as st
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Haversine distance у км
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

    mask = [True]  # перша точка завжди ок
    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i-1][[lat_col, lon_col]]
        lat2, lon2 = df.iloc[i][[lat_col, lon_col]]
        t1, t2 = df.iloc[i-1][time_col], df.iloc[i][time_col]
        dt = (t2 - t1).total_seconds() / 3600.0  # години

        if dt <= 0:
            mask.append(False)
            continue

        dist = haversine(lat1, lon1, lat2, lon2)  # км
        speed = dist / dt if dt > 0 else 0

        if speed > max_speed:
            print(f"Відсікаємо точку {i} (скачок {dist:.2f} км за {dt*60:.1f} хв со скоростью {speed:.1f} км/ч)") 
            mask.append(False)  # відсікаємо «скачок»
        else:
            mask.append(True)

    return df[mask].reset_index(drop=True)

st.set_page_config(layout="centered", page_icon="🚲", page_title="Eggrider Trip Analyzer")
st.title("🚲 Eggrider Trip Analyzer")

uploaded_file = st.file_uploader("Завантажте CSV з Eggrider", type=["csv"])

if uploaded_file:
    # читаємо CSV
    df = pd.read_csv(uploaded_file, sep=";", skiprows=1)

    min_dist = float(df["Distance(km)"].min())
    max_dist = float(df["Distance(km)"].max())

    dist_range = st.slider(
        "Виберіть діапазон дистанції (км)",
        min_value=min_dist,
        max_value=max_dist,
        value=(min_dist, max_dist),  # початково весь маршрут
        step=0.1
    )

    df = df[
        (df["Distance(km)"] >= dist_range[0]) &
        (df["Distance(km)"] <= dist_range[1])
    ]

    # створюємо вкладки
    tabs = st.tabs([
        "📊 Дані",
        "🛣️ Маршрут",
        "⚡ Швидкість та Потужність",
        "🔋 Напруга та Струм",
        "📈 Рівень підтримки",
    ])

    # ============= ДАННІ ==================
    with tabs[0]:
        st.subheader("📊 Дані")
        st.dataframe(df)

        # координати
        lat_col, lon_col = "Latitude", "Longitude"
        df = df.dropna(subset=[lat_col, lon_col])
        st.write(f"Знайдено {len(df)} координатних точок.")


    # ============= МАРШРУТ ==================
    with tabs[1]:
        st.subheader("🗺️ Маршрут на карті")
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

    # ============= ШВИДКІСТЬ + ПОТУЖНІСТЬ ==================
    with tabs[2]:
        st.subheader("⚡ Швидкість та Потужність")
        fig, ax1 = plt.subplots()

        ax1.plot(df["Distance(km)"], df["Speed(km/h)"], label="Speed (km/h)", color="blue")
        ax1.plot(df["Distance(km)"], df["SpeedGPS(km/h)"], label="GPS Speed (km/h)", color="green", alpha=0.6)
        ax1.set_ylabel("Швидкість (км/год)", color="blue")
        ax1.legend()


        ax2 = ax1.twinx()
        ax2.plot(df["Distance(km)"], df["MotorPower(W)"], label="Motor Power (W)", color="red", alpha=0.6)
        ax2.set_ylabel("Потужність (W)", color="red")

        ax1.set_xlabel("Дістанція (км)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ============= НАПРУГА / СТРУМ ==================
    with tabs[3]:
        st.subheader("🔋 Напруга та Струм")
        fig, ax1 = plt.subplots()

        ax1.plot(df["Distance(km)"], df["Voltage(V)"], color="orange", label="Voltage (V)")
        ax1.set_ylabel("Вольти (V)", color="orange")
        ax1.set_ylim(41, 54)

        ax2 = ax1.twinx()
        ax2.plot(df["Distance(km)"], df["Current(A)"], color="red", label="Current (A)", alpha=0.6)
        ax2.set_ylabel("Ампери (A)", color="red")

        ax1.set_xlabel("Дістанція (км)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ============= Рівень підтримки ==================
    with tabs[4]:
        st.subheader("📈 Рівень підтримки")
        fig, ax1 = plt.subplots()

        ax1.plot(df["Distance(km)"], df["AssistLevel"], label="PAS Level", color="brown")
        ax1.set_ylabel("Рівень підтримки", color="brown")
        ax1.set_ylim(0, 9)


        ax2 = ax1.twinx()
        ax2.plot(df["Distance(km)"], df["MotorPower(W)"], label="Motor Power (W)", color="red", alpha=0.6)
        ax2.set_ylabel("Потужність (W)", color="red")

        ax1.set_xlabel("Дістанція (км)")
        plt.xticks(rotation=45)
        st.pyplot(fig)