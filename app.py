import pandas as pd
import matplotlib.pyplot as plt
import folium, datetime
from streamlit_folium import st_folium
from folium.plugins import Fullscreen
import streamlit as st
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import altair as alt

st.set_page_config(layout="centered", page_icon="🚲", page_title="Eggrider Trip Analyzer")
st.title("🚲 Eggrider Trip Analyzer")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        margin: auto;
    }    
    @media (max-width: 640px) {
       #eggrider-trip-analyzer {
        font-size: 1.8rem;
       }
       }
    </style>
    """,
    unsafe_allow_html=True
)

def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))

def clean_gps_data(df, lat_col="Latitude", lon_col="Longitude", accuracy=10):
    mask = (
        df["AccuracyPosition(m)"] < accuracy
    )

    df_clean = df[mask].reset_index(drop=True)

    df_clean[lat_col] = df_clean[lat_col].rolling(window=5, center=True, min_periods=1).mean()
    df_clean[lon_col] = df_clean[lon_col].rolling(window=5, center=True, min_periods=1).mean()

    return df_clean

def reduce_points_by_distance(df, lat_col="Latitude", lon_col="Longitude", step_m=1):
    """
    Залишає одну точку кожні step_m метрів по маршруту.
    """
    from math import radians, cos, sin, atan2, sqrt

    if df.empty:
        return df

    keep_rows = [0]   # залишаємо першу точку
    dist_accum = 0.0
    last_lat, last_lon = df.iloc[0][lat_col], df.iloc[0][lon_col]

    for i in range(1, len(df)):
        lat, lon = df.iloc[i][lat_col], df.iloc[i][lon_col]
        d = haversine_km(last_lat, last_lon, lat, lon) * 1000
        dist_accum += d
        if dist_accum >= step_m:   # накопичили метр
            keep_rows.append(i)
            last_lat, last_lon = lat, lon
            dist_accum = 0.0

    # завжди додаємо останню точку
    if keep_rows[-1] != len(df) - 1:
        keep_rows.append(len(df) - 1)

    return df.iloc[keep_rows].reset_index(drop=True)

# ---------- helper to parse time ----------
def parse_time_column(df, time_col="Time(HH:mm:ss.fff)"):
    times = pd.to_timedelta(df[time_col])
    secs = times.dt.total_seconds().to_numpy()

    # handle day rollovers (00:00:00)
    #day_offset = 0
    #offsets = np.zeros(len(secs))
    #prev = secs[0]
    #for i in range(1, len(secs)):
    #    if secs[i] < prev:  # next day
    #        day_offset += 86400
    #    offsets[i] = day_offset
    #    prev = secs[i]
    offsets = 0

    base_date = pd.Timestamp("2000-01-01")
    ts = base_date + pd.to_timedelta(secs + offsets, unit="s")

    df = df.copy()
    df["timestamp"] = ts
    return df

def format_hms(hours: float) -> str:
    total_seconds = int(hours * 3600)
    return str(datetime.timedelta(seconds=total_seconds))

# ---------- Statistics tab ----------
def render_statistics(df):
    # ensure timestamp
    if "timestamp" not in df.columns:
        df = parse_time_column(df, "Time(HH:mm:ss.fff)")

    # compute dt
    df["dt_s"] = df["timestamp"].diff().dt.total_seconds().clip(lower=0).fillna(0)
    df["dt_h"] = df["dt_s"] / 3600

    total_distance = df["Distance(km)"].max() - df["Distance(km)"].min()

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

    # energy consumption
    total_ah, total_wh, avg_voltage = None, None, None
    if "Current(A)" in df.columns:
        total_ah = (df["Current(A)"] * df["dt_h"]).sum()
    if "MotorPower(W)" in df.columns:
        total_wh = (df["MotorPower(W)"] * df["dt_h"]).sum()
    if "Voltage(V)" in df.columns:
        avg_voltage = df["Voltage(V)"].mean()
        max_voltage = df["Voltage(V)"].max()
        min_voltage = df["Voltage(V)"].min()

    wh_per_km = (total_wh / total_distance) if total_wh and total_distance > 0 else None
    
    # battery usage
    start_bettery = df["BatteryPercentage"].iloc[0] if "BatteryPercentage" in df.columns else None
    end_battery = df["BatteryPercentage"].iloc[-1] if "BatteryPercentage" in df.columns else None
    total_battery_percent = df["BatteryPercentage"].max() - df["BatteryPercentage"].min() if "BatteryPercentage" in df.columns else None 
    battery_per_km = (total_battery_percent / total_distance) if total_battery_percent and total_distance > 0 else None

    min_temp = df["DisplayTemp(C)"].min()
    max_temp = df["DisplayTemp(C)"].max()
    avg_temp = df["DisplayTemp(C)"].mean()

    alt_min = df["Altitude(m)"].where(df["Altitude(m)"] > 0).min()
    alt_max = df["Altitude(m)"].max()
    alt_diff = alt_max - alt_min

    df["DistancePercentage"] = df["Distance(km)"].diff().clip(lower=0).fillna(0)

    st.markdown("""
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 20px;
    }
    @media (max-width: 640px) {
        .stColumn {
           width: calc(49.6667% - 1rem);
           flex: 1 1 calc(49.6667% - 1rem);
           min-width: auto;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    with col1:
        st.metric("Total Distance", f"{total_distance:.2f} km")
        st.metric("Total Time", f"{format_hms(total_time)}")
        st.metric("Moving Time", f"{format_hms(moving_time)}")
    with col2:
        if max_speed is not None:
            st.metric("Max Speed", f"{max_speed:.1f} km/h")
            st.metric("Avg Speed", f"{avg_speed:.1f} km/h")
    with col3:
        if max_power is not None:
            st.metric("Max Motor Power", f"{max_power:.0f} W")
            st.metric("Avg Motor Power", f"{avg_power:.0f} W")
    with col4:
        if max_current is not None:
            st.metric("Max Current", f"{max_current:.1f} A")  
        if avg_current is not None:
            st.metric("Avg Current", f"{avg_current:.1f} A")
    with col5:
        if max_voltage:
            st.metric("Max Voltage", f"{max_voltage:.1f} V")
        if min_voltage is not None:
            st.metric("Min Voltage", f"{min_voltage:.1f} V")
        if avg_voltage is not None:
            st.metric("Avg Voltage", f"{avg_voltage:.1f} V")
    with col6:
        if start_bettery is not None and end_battery is not None:
            st.metric("Start Battery", f"{start_bettery:.0f} %")
            st.metric("End Battery", f"{end_battery:.0f} %")  
        if total_battery_percent is not None:
            st.metric("Battery Used", f"{total_battery_percent:.1f} %")
    with col7:
        if min_temp:
            st.metric("Min Temperature", f"{min_temp:.1f} °C")
        if max_temp:
            st.metric("Max Temperature", f"{max_temp:.1f} °C")
        if avg_temp:
            st.metric("Avg Temperature", f"{avg_temp:.1f} °C")
    with col8:
        if alt_min:
            st.metric("Altitude Min", f"{alt_min:.1f} m")
        if alt_max:
            st.metric("Altitude Max", f"{alt_max:.1f} m")
        if alt_diff:
            st.metric("Altitude Diff", f"{alt_diff:.1f} m")


    # діапазони швидкості (можна змінювати під свої потреби)
    speed_bins = [-0.1, 0, 5, 10, 20, 30, 40, 50, 100]  
    speed_labels = [
        "0 km/h",
        "1-5 km/h",
        "6-10 km/h",
        "11-20 km/h",
        "21-30 km/h",
        "31-40 km/h",
        "41-50 km/h",
        ">50 km/h"
    ]

    df["SpeedRange"] = pd.cut(df["Speed(km/h)"], bins=speed_bins, labels=speed_labels, right=True)

    # скільки км в кожному діапазоні
    dist_by_speed = df.groupby("SpeedRange")["DistancePercentage"].sum()

    # у відсотках
    dist_percent_speed = dist_by_speed / dist_by_speed.sum() * 100

    st.subheader("🚴 Speed (%)")

    # графік
    st.bar_chart(dist_percent_speed)

    st.subheader("⚙️ Motor Power usage (%)")

    # сумарний пробіг з мотором та без
    power_distance = {
        "0 W": df.loc[df["MotorPower(W)"] <= 0, "DistancePercentage"].sum(),
        ">0 W": df.loc[df["MotorPower(W)"] > 0, "DistancePercentage"].sum()
    }



    # знайти максимум потужності
    max_power = df["MotorPower(W)"].max()

    # базові бінові межі
    bins = [-0.1, 0, 100, 200, 300, 500, 1000, 1200]

    # додаємо верхній ліміт, якщо він більший за останній
    if max_power > bins[-1]:
        bins.append(max_power)

    labels = [
        "0 W",
        "1-100 W",
        "101-200 W",
        "201-300 W",
        "301-500 W",
        "501-1000 W",
        "1001-1200 W",
    ]

    if max_power > 1200:
        labels.append(f">1200-{int(max_power)} W")

    # категоризація з параметром duplicates="drop"
    df["PowerRange"] = pd.cut(
        df["MotorPower(W)"], 
        bins=bins, 
        labels=labels, 
        right=True, 
        duplicates="drop"
    )

    dist_by_power = df.groupby("PowerRange")["DistancePercentage"].sum()
    dist_percent = dist_by_power / dist_by_power.sum() * 100

    st.bar_chart(dist_percent)

    # у відсотках
    total_dist = sum(power_distance.values())
    power_percent = {k: (v / total_dist * 100 if total_dist > 0 else 0)
                    for k, v in power_distance.items()}

    st.write(f"Distance without Motor assist: **{power_distance['0 W']:.2f} km ({power_percent['0 W']:.1f}%)**")
    st.write(f"Distance with Motor assist: **{power_distance['>0 W']:.2f} km ({power_percent['>0 W']:.1f}%)**")

    st.subheader("💪 Assist Usage (%)")
    # assist distribution
    assist_percent = df["AssistLevel"].value_counts(normalize=True).sort_index() * 100
    st.bar_chart(assist_percent)

    #dist_by_power = df.groupby("PowerRange")["DistancePercentage"].sum()
    #dist_percent = dist_by_power / dist_by_power.sum() * 100  


    st.subheader("🔋 Energy Consumption")
    if battery_per_km is not None:
        st.write(f"Battery Usage: **{battery_per_km:.2f} %/km**")
    if total_ah is not None:
        st.write(f"Consumed Charge: **{total_ah:.2f} Ah**")
    if total_wh is not None:
        st.write(f"Consumed Energy: **{total_wh:.1f} Wh**")
    if wh_per_km is not None:
        st.write(f"Specific Consumption: **{wh_per_km:.1f} Wh/km**")

def fix_incorect_distance(df, dist_col="Distance(km)"):
    corrected = []
    offset = 0
    prev_raw = df[dist_col].iloc[0]
    prev_corr = df[dist_col].iloc[0]

    for val in df[dist_col]:
        # новий сегмент (Distance скинувся)
        if val < prev_raw:
            offset += prev_corr

        # кориговане значення
        new_val = val + offset

        # видалення локальних максимумів: якщо зменшилось → тягнемо попереднє
        if new_val < prev_corr:
            new_val = prev_corr

        corrected.append(new_val)
        prev_raw = val
        prev_corr = new_val

    df["Distance(km)"] = corrected
    return df    

def fix_multi_df(df, dist_col="Distance(km)"):
    #df = df.sort_values("Time(HH:mm:ss.fff)").reset_index(drop=True)

    corrected = []
    offset = 0
    prev = df[dist_col].iloc[0]

    for val in df[dist_col]:
        # якщо значення зменшилось → це новий сегмент
        if val < prev:
            offset += prev  # додаємо максимум попереднього сегменту
        corrected.append(val + offset)
        prev = val

    df["Distance(km)"] = corrected
    return df

# ----------------------------
# UI
# ----------------------------

uploaded_files = st.file_uploader("Upload CSV from Eggrider", type=["csv"], accept_multiple_files=True)

dfs = []

if uploaded_files:
    uploaded_files = sorted(uploaded_files, key=lambda x: x.name)
    # якщо користувач завантажив свій файл
    for f in uploaded_files:
        try:
            fl = f.readline().decode("utf-8").strip()
            if "sep=;" in fl:
                d = pd.read_csv(f, sep=";")
            else:
                f.seek(0)
                d = pd.read_csv(f)
            dfs.append(d)
        except Exception as e:
            pass
        
if dfs:
    df = fix_multi_df(pd.concat(dfs, ignore_index=True))
else:
    # fallback: завантажуємо демо-дані з диску
    demo_path = "demo.csv"   # шлях до вашого demo-файлу
    df = pd.read_csv(demo_path, sep=";", skiprows=1)
    st.info("ℹ️ Using demo.csv. Upload your CSV from Eggrider.")

if not df.empty:
    min_dist = float(df["Distance(km)"].min())
    max_dist = float(df["Distance(km)"].max())

    dist_range = st.slider(
        "🛤️ Select distance range (km)",
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
        "📈 Chart",
        "🗺️ Map",
        "𝄜 Data",
    ])

    # ============= STAT ==================

    with tabs[0]:
        st.header("📊 Ride Statistics")
        render_statistics(df)

    # ============= DATA ==================
    with tabs[3]:
        st.header("𝄜 Raw Data")
        st.dataframe(df, height=780)
        st.write(f"Found {len(df)} records.")    

    # ============= MAP ==================
    with tabs[2]:
        st.header("🗺️ Route on map")
        df = reduce_points_by_distance(clean_gps_data(df))
        if not df.empty:

            marker_range = st.slider(
                "⏺ Select distance between markers (m)",
                min_value=0,
                max_value=1000,
                value=10,
                step=10
            )

            start_coords = (df["Latitude"].iloc[0], df["Longitude"].iloc[0])
            trip_map = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=13)

            coords = df[["Latitude", "Longitude"]].values.tolist()
            folium.PolyLine(coords, color="blue", weight=3).add_to(trip_map)
            folium.Marker(coords[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(trip_map)

            if marker_range > 0:
                distance_accum = 0
                next_marker = 0  
                for i in range(1, len(coords)):
                    lat1, lon1 = coords[i-1]
                    lat2, lon2 = coords[i]
                    dist = haversine_km(lat1, lon1, lat2, lon2) * 1000  # відстань між точками
                    distance_accum += dist
                    if distance_accum >= next_marker:
                        row = df.iloc[i]
                        tooltip_text = (
                            f"📍 {next_marker} m<br>"
                            f"🚴 Speed: {row['Speed(km/h)']:.1f} km/h<br>"
                            f"⚡ Motor Power: {row['MotorPower(W)']:.0f} W<br>"
                            f"🔋 Voltage: {row['Voltage(V)']:.1f} V<br>"
                            f"🔋 Battery: {row['BatteryPercentage']:.0f}%<br>"
                            f"🤖 Assist: {row['AssistLevel']}<br>"
                            f"🚙 OffRoad Mode: {row['OffRoadMode']}"
                        )
                        radius = (int(row["AssistLevel"]) / 4) + 1 + (int(row["OffRoadMode"] * 2))
                        if row["MotorPower(W)"] > 0: 
                           if row["OffRoadMode"] == 1:
                               color = "brown" 
                           else:
                               color = "red"
                        else:
                            color = "green"
                        folium.CircleMarker([lat2, lon2],
                                    tooltip=tooltip_text, radius=radius, color=color, fill=True, fill_opacity=0.8).add_to(trip_map)
                        next_marker += marker_range  

            folium.Marker(coords[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(trip_map)

            Fullscreen(
                position="topright",      # позиція кнопки (topright, topleft, bottomright, bottomleft)
                title="Open full screen", # текст підказки
                title_cancel="Exit full screen", 
                force_separate_button=True
            ).add_to(trip_map)

            st_folium(trip_map, width="100%", returned_objects=[])
            st.write(f"Found {len(df)} location points.")

    # ============= CHART ==================
    with tabs[1]:
        st.header("📈 Chart")

        df["MotorPower(W/10)"] = df["MotorPower(W)"] / 10.0
        df["AssistLevel"] = df["AssistLevel"] * 10
        df["OffRoadMode"] = df["OffRoadMode"] * 100

        cols_available = [
        "Speed(km/h)", 
        "SpeedGPS(km/h)", 
        "MotorPower(W/10)", 
        "Current(A)", 
        "Voltage(V)", 
        "BatteryPercentage", 
        "AssistLevel",
        "OffRoadMode",
        "DisplayTemp(C)"
        ]

        def_cols =[
        "Speed(km/h)", 
        "MotorPower(W/10)", 
        "BatteryPercentage", 
        "AssistLevel"
        ]

        selected_cols = []
        with st.expander("⚙️ Select parameters to plot"):
            for c in cols_available:
                if st.checkbox(c, value=(c in def_cols)):
                    selected_cols.append(c)

        if selected_cols:
            # перетворюємо у long формат для Altair
            df_long = df.melt(id_vars=["Distance(km)"], 
                            value_vars=selected_cols, 
                            var_name="Parameter", 
                            value_name="Value")
            
            chart = (
                alt.Chart(df_long)
                .mark_line()
                .encode(
                    x=alt.X("Distance(km):Q", title="Distance (km)"),
                    y=alt.Y("Value:Q"),
                    color="Parameter:N",
                    tooltip=["Distance(km)", "Parameter", "Value"]
                )
                .interactive() 
                .properties(
                    height=800
                )
            )

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("☝️ Select at least one parameter to display chart")



st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        color: #999;
        text-align: center;
        padding: 5px;
        font-size: 11px;
        background: rgb(14, 17, 23);
    }
    a {
        color: #bbb !important;
        text-decoration: none !important;
    }
    </style>
    <div class="footer">
       2025 © Powered by <a href='mailto:silentlexx@gmail.com'>Silentlexx</a>. v1.6
    </div>
    """,
    unsafe_allow_html=True
)