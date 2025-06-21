"""
Streamlit Vehicle Heatmap Dashboard
Interactive web application for vehicle trajectory visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from branca.element import MacroElement, Template
import os

# Configure Streamlit page
st.set_page_config(page_title="Vehicle Heatmap Enhanced", layout="wide")
st.title("🚚 Vehicle Heatmap Dashboard (Enhanced Streamlit Version)")

# Initialize session state
if "filters_applied" not in st.session_state:
    st.session_state["filters_applied"] = False
if "manual_center" not in st.session_state:
    st.session_state["manual_center"] = ""

# Data file configuration
DATA_DIR = "Data"
CSV_FILE = os.path.join(DATA_DIR, "cleaned_Kolkta_2_spots.csv")

# Check if data file exists
if not os.path.exists(CSV_FILE):
    st.error(f"❌ Data file not found: {CSV_FILE}")
    st.info("Please ensure the data file is in the Data/ directory")
    st.stop()

@st.cache_data
def load_and_process_data(csv_file):
    """Load and preprocess the GPS data"""
    progress = st.progress(0, text="🔄 Loading data...")
    
    usecols = ['deviceId', 'gpsTime', 'latitude', 'longitude', 'deviceSpeed']
    dtype_map = {
        'deviceId': 'category',
        'latitude': 'float32',
        'longitude': 'float32',
        'deviceSpeed': 'float32'
    }
    
    reader = pd.read_csv(
        csv_file, 
        usecols=usecols, 
        dtype=dtype_map,
        parse_dates=['gpsTime'], 
        date_format="%Y-%m-%d %H:%M:%S", 
        chunksize=200_000
    )
    
    filtered_chunks = []
    total_rows = 0
    
    for i, chunk in enumerate(reader):
        # Filter for target area
        chunk = chunk[
            (chunk['latitude'] >= 22.50) & (chunk['latitude'] <= 22.54) &
            (chunk['longitude'] >= 88.29) & (chunk['longitude'] <= 88.32)
        ].copy()
        
        if not chunk.empty:
            chunk['deviceId'] = chunk['deviceId'].astype(str)
            chunk['hour'] = chunk['gpsTime'].dt.hour
            filtered_chunks.append(chunk)
            total_rows += len(chunk)
        
        progress.progress(min(40, 10 + i * 5), text=f"📦 Processing chunk {i+1}...")
    
    df = pd.concat(filtered_chunks, ignore_index=True).sort_values(['deviceId', 'gpsTime'])
    progress.progress(100, text=f"✅ Loaded {total_rows:,} filtered rows.")
    
    return df

def detect_io_patterns(df):
    """Detect inbound/outbound patterns for each device"""
    def detect_io(group):
        group['io_flag'] = 'OUT'
        if not group.empty:
            group.loc[group.index[0], 'io_flag'] = 'INBOUND'
            group.loc[group.index[-1], 'io_flag'] = 'OUTBOUND'
        return group
    
    return df.groupby('deviceId', group_keys=False).apply(detect_io)

def detect_stay_points(df):
    """Detect and analyze stay points"""
    # Detect stationary points
    df['lat_shift'] = df.groupby('deviceId')['latitude'].shift()
    df['lon_shift'] = df.groupby('deviceId')['longitude'].shift()
    df['is_stationary'] = (
        (np.isclose(df['latitude'], df['lat_shift'], atol=1e-5)) &
        (np.isclose(df['longitude'], df['lon_shift'], atol=1e-5)) &
        (df['deviceSpeed'] == 0)
    )
    df['stationary_group'] = (df['is_stationary'] != df['is_stationary'].shift()).cumsum()
    
    # Summarize stay points
    stay_summary = df[df['is_stationary']].groupby(['deviceId', 'stationary_group']).agg({
        'gpsTime': ['first', 'last'],
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()
    
    stay_summary.columns = ['deviceId', 'stationary_group', 'start_time', 'end_time', 'latitude', 'longitude']
    stay_summary['duration_min'] = (stay_summary['end_time'] - stay_summary['start_time']).dt.total_seconds() / 60
    
    def classify_color(duration_min):
        """Classify stay duration into color categories"""
        if duration_min <= 30:
            return 'green'
        elif duration_min <= 60:
            return 'blue'
        elif duration_min <= 180:
            return 'yellow'
        elif duration_min <= 300:
            return 'orange'
        else:
            return 'red'
    
    stay_summary['color'] = stay_summary['duration_min'].apply(classify_color)
    stay_summary['popup'] = stay_summary['duration_min'].apply(lambda x: f"{x:.1f} min")
    
    return stay_summary

def create_legend():
    """Create map legend for stay duration colors"""
    legend_template = """
    {% macro html(this, kwargs) %}
    <div style="
        position: absolute; 
        bottom: 30px; left: 30px; z-index: 9999; 
        background-color: white; 
        padding: 10px; 
        border: 2px solid grey; 
        border-radius: 5px;
        font-size:14px;
    ">
    <b>🖍 Legend - Stop Duration</b><br>
    <i style='color:green;'>●</i> ≤30min<br>
    <i style='color:blue;'>●</i> ≤1hr<br>
    <i style='color:yellow;'>●</i> ≤3hr<br>
    <i style='color:orange;'>●</i> ≤5hr<br>
    <i style='color:red;'>●</i> >5hr
    </div>
    {% endmacro %}
    """
    
    legend = MacroElement()
    legend._template = Template(legend_template)
    return legend

# Main application
def main():
    # Load data
    df = load_and_process_data(CSV_FILE)
    
    # Get unique values for filters
    device_ids = sorted(df['deviceId'].unique())
    hours = sorted(df['hour'].unique())
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        selected_devices = st.multiselect(
            "Select Device ID(s)", 
            ["All"] + device_ids, 
            default=["All"]
        )
        selected_hours = st.multiselect(
            "Select Hour(s)", 
            ["All"] + [str(h) for h in hours], 
            default=["All"]
        )
        st.session_state["manual_center"] = st.text_input(
            "📍 Center Map on Lat,Lon", 
            value=st.session_state["manual_center"],
            help="Format: lat,lon (e.g., 22.52,88.30)"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            apply_clicked = st.button("✅ Apply Filters", use_container_width=True)
        with col2:
            reset_clicked = st.button("🔁 Reset", use_container_width=True)
    
    # Handle button clicks
    if apply_clicked:
        st.session_state["filters_applied"] = True
    if reset_clicked:
        st.session_state["filters_applied"] = False
        st.session_state["manual_center"] = ""
        st.experimental_rerun()
    
    if not st.session_state["filters_applied"]:
        st.warning("⏳ Please apply filters to render the map", icon="⏳")
        st.info("Use the sidebar controls to select devices and time periods, then click 'Apply Filters'")
        return
    
    # Apply filters
    filtered_df = df.copy()
    if "All" not in selected_devices:
        filtered_df = filtered_df[filtered_df['deviceId'].isin(selected_devices)]
    if "All" not in selected_hours:
        filtered_df = filtered_df[filtered_df['hour'].isin([int(h) for h in selected_hours])]
    
    if filtered_df.empty:
        st.error("❌ No data found with the selected filters")
        return
    
    # Process data
    with st.spinner("🔄 Processing data..."):
        filtered_df = detect_io_patterns(filtered_df)
        stay_summary = detect_stay_points(filtered_df)
    
    # Determine map center
    try:
        if st.session_state["manual_center"]:
            lat_str, lon_str = st.session_state["manual_center"].split(",")
            center_lat, center_lon = float(lat_str.strip()), float(lon_str.strip())
        else:
            center_lat = filtered_df['latitude'].mean()
            center_lon = filtered_df['longitude'].mean()
    except:
        center_lat = filtered_df['latitude'].mean()
        center_lon = filtered_df['longitude'].mean()
        if st.session_state["manual_center"]:
            st.warning("⚠️ Invalid center coordinates format. Using data center.")
    
    # Create map
    with st.spinner("🗺️ Creating map..."):
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
        
        # Create feature groups for different layers
        fg_heat = folium.FeatureGroup(name="Heat Map").add_to(m)
        fg_stay = folium.FeatureGroup(name="Stay Points").add_to(m)
        fg_trace = folium.FeatureGroup(name="Trajectory").add_to(m)
        
        # Add heatmap layer
        heat_points = [
            [row['latitude'], row['longitude'], min(20 / max(row['deviceSpeed'], 0.1), 20)] 
            for _, row in filtered_df.iterrows() if row['deviceSpeed'] > 0
        ]
        if heat_points:
            HeatMap(heat_points, min_opacity=0.3, max_zoom=18, radius=12, blur=8).add_to(fg_heat)
        
        # Add stay points
        for _, row in stay_summary.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=row['color'],
                fill=True,
                fill_opacity=0.9,
                popup=f"Device: {row['deviceId']}<br>Duration: {row['popup']}"
            ).add_to(fg_stay)
        
        # Add trajectory lines (simplified)
        for device_id, group in filtered_df.groupby('deviceId'):
            group = group.reset_index(drop=True)
            for i in range(0, len(group)-2, 3):  # Sample every 3rd point for performance
                coords = list(zip(
                    group.loc[i:i+2, 'latitude'], 
                    group.loc[i:i+2, 'longitude']
                ))
                if len(coords) >= 2:
                    folium.PolyLine(
                        coords, 
                        color="gray", 
                        weight=2, 
                        opacity=0.6,
                        popup=f"Device: {device_id}"
                    ).add_to(fg_trace)
        
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Add legend
        legend = create_legend()
        m.get_root().add_child(legend)
    
    # Display map
    st.subheader("🗺️ Interactive Vehicle Heatmap")
    map_data = st_folium(m, width=1100, height=700)
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Records", f"{len(filtered_df):,}")
    with col2:
        st.metric("🚛 Unique Vehicles", len(filtered_df['deviceId'].unique()))
    with col3:
        st.metric("🛑 Stay Points", len(stay_summary))
    with col4:
        avg_speed = filtered_df[filtered_df['deviceSpeed'] > 0]['deviceSpeed'].mean()
        st.metric("⚡ Avg Speed", f"{avg_speed:.1f} km/h" if not pd.isna(avg_speed) else "N/A")
    
    # Additional legend in sidebar
    with st.sidebar:
        st.markdown("""
        ### 🖍️ Legend
        **Stop Duration Colors:**
        - 🟢 ≤30 min
        - 🔵 ≤1 hr  
        - 🟡 ≤3 hr
        - 🟠 ≤5 hr
        - 🔴 >5 hr
        
        **Map Layers:**
        - **Heat Map**: Traffic intensity
        - **Stay Points**: Parking/stop locations
        - **Trajectory**: Vehicle paths
        """)

if __name__ == "__main__":
    main()