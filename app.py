import pandas as pd
import matplotlib.pyplot as plt
import folium, datetime
import streamlit as st
import numpy as np
from streamlit.components.v1 import html
from math import radians, sin, cos, sqrt, atan2

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

def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))

def clean_gps_data(df, lat_col="Latitude", lon_col="Longitude", speed_col="Speed(km/h)", time_col="Time(HH:mm:ss.fff)"):
    df = df.copy()
  
    max_speed_kmh = df[speed_col].max()
    max_speed_mps = max_speed_kmh / 3.6  # convert to m/s
    
    # 1. Час → timedelta (секунди від початку доби)
    times = pd.to_timedelta(df[time_col])
    secs = times.dt.total_seconds().to_numpy()

    # 2. Обробка переходів через північ
    day_offset = 0
    offsets = np.zeros(len(secs))
    prev = secs[0]
    for i in range(1, len(secs)):
        if secs[i] < prev:
            day_offset += 86400
        offsets[i] = day_offset
        prev = secs[i]

    df["time_s"] = secs + offsets

    # 3. Відстань між точками (метри)
    lats = df[lat_col].to_numpy()
    lons = df[lon_col].to_numpy()
    ts = df["time_s"].to_numpy()
    dists = [0]
    delta_s = [0]
    for i in range(1, len(df)):
       try:
            dists.append(haversine_km(lats[i-1], lons[i-1], lats[i], lons[i]) * 1000)
            delta_s.append(ts[i] - ts[i-1])
       except:
            dists.append(0)
            delta_s.append(0)
    df["dist_m"] = dists
    df["dt_s"] = delta_s

    max_delta_s = df["dt_s"].mean() 
    max_jump_m = max_speed_mps * max_delta_s * 50 # FIXME: 50 - empirical factor
    if max_jump_m < 100:
        max_jump_m = 100

    # 5. Маска:
    mask = (
        #(df[speed_col] <= max_speed_kmh) &      # не нереальна швидкість
        (df["dist_m"] > 0) &
        (df["dist_m"] <= max_jump_m)             # не великий стрибок
        #(df[speed_col] > 1) 
        #~((df["dist_m"] < min_move_m) & (df["speed_kmh"] < min_speed_kmh)) # прибираємо "тремтіння"
    )

    df_clean = df[mask].reset_index(drop=True)

    # 6. Згладжування (ковзне середнє на 5 точок)
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
        max_voltage = df["Voltage(V)"].max()
        min_voltage = df["Voltage(V)"].min()

    wh_per_km = (total_wh / total_distance) if total_wh and total_distance > 0 else None
    
    # battery usage
    start_bettery = df["BatteryPercentage"].iloc[0] if "BatteryPercentage" in df.columns else None
    end_battery = df["BatteryPercentage"].iloc[-1] if "BatteryPercentage" in df.columns else None
    total_battery_percent = df["BatteryPercentage"].max() - df["BatteryPercentage"].min() if "BatteryPercentage" in df.columns else None 
    battery_per_km = (total_battery_percent / total_distance) if total_battery_percent and total_distance > 0 else None

    df["dist_km"] = df["Distance(km)"].diff().clip(lower=0).fillna(0)

    st.markdown("""
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
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

    st.subheader("⚙️ Motor Power usage")

    # сумарний пробіг з мотором та без
    power_distance = {
        "0 W": df.loc[df["MotorPower(W)"] <= 0, "dist_km"].sum(),
        ">0 W": df.loc[df["MotorPower(W)"] > 0, "dist_km"].sum()
    }

    # у відсотках
    total_dist = sum(power_distance.values())
    power_percent = {k: (v / total_dist * 100 if total_dist > 0 else 0)
                    for k, v in power_distance.items()}

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"Distance without Motor assist:")
        st.write(f"Distance with Motor assist:")

    with col2:
        st.write(f"**{power_distance['0 W']:.2f} km**")
        st.write(f"**{power_distance['>0 W']:.2f} km**")

    with col3:
        st.write(f"**{power_percent['0 W']:.1f}%**")
        st.write(f"**{power_percent['>0 W']:.1f}%**")

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

    if max_power > 1000:
        labels.append(f">1000-{int(max_power)} W")

    # категоризація з параметром duplicates="drop"
    df["PowerRange"] = pd.cut(
        df["MotorPower(W)"], 
        bins=bins, 
        labels=labels, 
        right=True, 
        duplicates="drop"
    )

    dist_by_power = df.groupby("PowerRange")["dist_km"].sum()
    dist_percent = dist_by_power / dist_by_power.sum() * 100

    st.bar_chart(dist_percent)

    if assist_percent is not None:
        st.subheader("💪 Assist Usage (%)")
        st.bar_chart(assist_percent)

    st.subheader("🔋 Energy Consumption")
    if battery_per_km is not None:
        st.write(f"Battery Usage: **{battery_per_km:.2f} %/km**")
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
        "𝄜 Data",
        "🗺️ Route",
        "⚡ Speed & Power",
        "🔋 Voltage & Current",
        "💪 Assist & Power",
        "🚀 Assist & Speed",
    ])

    # ============= STAT ==================

    with tabs[0]:
        st.subheader("📊 Ride Statistics")
        render_statistics(df)

    # ============= DATA ==================
    with tabs[1]:
        st.subheader("𝄜 Raw Data")
        st.dataframe(df)
        st.write(f"Found {len(df)} records.")    

    # ============= ROUTE ==================
    with tabs[2]:
        st.subheader("🗺️ Route on map")
        dfg = reduce_points_by_distance(clean_gps_data(df))
        if not dfg.empty:

            marker_range = st.slider(
                "⏺ Select distance between markers (m)",
                min_value=0,
                max_value=1000,
                value=10,
                step=10
            )

            start_coords = (dfg["Latitude"].iloc[0], dfg["Longitude"].iloc[0])
            trip_map = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=13)

            coords = dfg[["Latitude", "Longitude"]].values.tolist()
            folium.PolyLine(coords, color="blue", weight=3).add_to(trip_map)
            folium.Marker(coords[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(trip_map)

            if marker_range > 0:
                # --- додавання маркерів кожні 100 м ---
                distance_accum = 0
                next_marker = 100  # перший маркер через 100 м
                for i in range(1, len(coords)):
                    lat1, lon1 = coords[i-1]
                    lat2, lon2 = coords[i]
                    dist = haversine_km(lat1, lon1, lat2, lon2) * 1000  # відстань між точками
                    distance_accum += dist
                    if distance_accum >= next_marker:
                        row = dfg.iloc[i]
                        tooltip_text = (
                            f"📍 {next_marker} m<br>"
                            f"🚴 Speed: {row['Speed(km/h)']:.1f} km/h<br>"
                            f"⚡ Motor Power: {row['MotorPower(W)']:.0f} W<br>"
                            f"🔋 Voltage: {row['Voltage(V)']:.1f} V<br>"
                            f"🔋 Battery: {row['BatteryPercentage']:.0f}%<br>"
                            f"🤖 Assist: {row['AssistLevel']}"
                        )
                        radius = (int(row["AssistLevel"]) / 4) + 1
                        if row["MotorPower(W)"] > 0: 
                            color = "red" 
                        else:
                            color = "green"
                        folium.CircleMarker([lat2, lon2],
                                    tooltip=tooltip_text, radius=radius, color=color, fill=True, fill_opacity=0.8).add_to(trip_map)
                        next_marker += marker_range  # наступна ціль через 100 м

            folium.Marker(coords[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(trip_map)

            html(fullscreen_html(trip_map._repr_html_()), height=400)
            st.write(f"Found {len(dfg)} location points.")
            #st.dataframe(dfg)

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

    # ============= ASSIST / POWER ==================
    with tabs[5]:
        st.subheader("💪 Assist Level & Power")
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

    # ============= ASSIST / SPEED ==================
    with tabs[6]:
        st.subheader("🚀 Assist Level & Speed")
        fig, ax1 = plt.subplots()

        ax1.plot(df["Distance(km)"], df["AssistLevel"], label="PAS Level", color="brown")
        ax1.set_ylabel("Assist Level", color="brown")
        ax1.set_ylim(0, 9)


        ax2 = ax1.twinx()
        ax2.plot(df["Distance(km)"], df["Speed(km/h)"], label="Motor Power (W)", color="blue", alpha=0.9)
        ax2.set_ylabel("Speed (km/h)", color="blue")

        ax1.set_xlabel("Distance (km)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    #st.html("<div>Powered by silentlexx. V.1.1</div>")
