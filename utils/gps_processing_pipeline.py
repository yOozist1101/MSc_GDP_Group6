# gps_processing_pipeline.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from geopy.distance import geodesic
import folium

# ========== Utility ==========
def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat, dlon = lat2 - lat1, np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# ========== Step 1: File Concatenation ==========
def concat_files(file_list, output_file, chunksize=100000):
    with open(output_file, 'w', encoding='utf-8') as f_out:
        first = True
        for file in tqdm(file_list, desc="Merging files"):
            for chunk in pd.read_csv(file, chunksize=chunksize):
                chunk.to_csv(f_out, header=first, index=False, mode='a')
                first = False

# ========== Step 2: Match POI ==========
def match_trajectories_to_poi(trajectory_file, poi_file, output_file, radius_meters=5000):
    pois = pd.read_csv(poi_file)
    device_to_target = {}
    for chunk in pd.read_csv(trajectory_file, usecols=['deviceId', 'latitude', 'longitude'], chunksize=10**6):
        for _, poi in pois.iterrows():
            dist = haversine_np(chunk['latitude'].values, chunk['longitude'].values, poi['latitude'], poi['longitude'])
            matched = chunk.loc[dist <= radius_meters, 'deviceId']
            for device_id in matched:
                if device_id not in device_to_target:
                    device_to_target[device_id] = poi['Target']
    first_write = True
    for chunk in pd.read_csv(trajectory_file, chunksize=10**6):
        matched = chunk[chunk['deviceId'].isin(device_to_target)]
        if not matched.empty:
            matched['Target'] = matched['deviceId'].map(device_to_target)
            matched.to_csv(output_file, mode='a', index=False, header=first_write)
            first_write = False

# ========== Step 3: Clean GPS ==========
def clean_gps_data(df):
    df = df.dropna(subset=['latitude', 'longitude', 'gpsTime', 'deviceId'])
    df = df[(df['latitude'].between(6.0, 37.5)) & (df['longitude'].between(68.0, 97.5))]
    df = df[(df['deviceSpeed'] >= 0) & (df['deviceSpeed'] <= 150)]
    df['gpsTime'] = pd.to_datetime(df['gpsTime'], unit='ms', errors='coerce')
    df = df.dropna(subset=['gpsTime'])
    df['latitude'] = df['latitude'].round(6)
    df['longitude'] = df['longitude'].round(6)
    return df

def remove_jump_points(group):
    if len(group) < 2:
        return group
    group = group.sort_values('gpsTime')
    cleaned = [group.iloc[0]]
    for i in range(1, len(group)):
        prev, curr = cleaned[-1], group.iloc[i]
        dist = geodesic((prev['latitude'], prev['longitude']), (curr['latitude'], curr['longitude'])).meters
        time_diff = (curr['gpsTime'] - prev['gpsTime']).total_seconds()
        speed = (dist / 1000) / (time_diff / 3600) if time_diff > 0 else 0
        if dist <= 10000 and time_diff <= 1800:
            cleaned.append(curr)
    return pd.DataFrame(cleaned)

def process_and_clean_gps(input_file, output_file):
    chunks = pd.read_csv(input_file, chunksize=100000)
    cleaned_chunks = []
    for chunk in chunks:
        clean_chunk = clean_gps_data(chunk)
        grouped = clean_chunk.groupby('deviceId', group_keys=False)
        cleaned_chunks.append(grouped.apply(remove_jump_points))
    final_df = pd.concat(cleaned_chunks)
    if 'vehicleClass' in final_df.columns:
        final_df = final_df[final_df['vehicleClass'] == 3]
    final_df.to_csv(output_file, index=False)

# ========== Step 4: Visualize ==========
def visualize_trajectories(trajectory_file, poi_file, output_html, max_vehicles=300):
    df = pd.read_csv(trajectory_file)
    pois = pd.read_csv(poi_file)
    if 'gpsTime' in df.columns:
        try:
            sample = df['gpsTime'].iloc[0]
            if isinstance(sample, (int, float, np.integer, np.float64)):
                df['gpsTime'] = pd.to_datetime(df['gpsTime'] / 1000, unit='s')
            else:
                df['gpsTime'] = pd.to_datetime(df['gpsTime'], errors='coerce')
        except:
            pass
    df = df.sort_values(by=['deviceId', 'gpsTime'] if 'gpsTime' in df.columns else ['deviceId'])
    selected_ids = df['deviceId'].unique()[:max_vehicles]
    df_subset = df[df['deviceId'].isin(selected_ids)]
    m = folium.Map(location=[pois.iloc[0]['latitude'], pois.iloc[0]['longitude']], zoom_start=10)
    for _, row in pois.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=row['Target'],
            icon=folium.Icon(color='red', icon='star', prefix='fa')
        ).add_to(m)
    colors = ['blue', 'green', 'purple', 'orange', 'black', 'cadetblue', 'darkred']
    for i, device_id in enumerate(selected_ids):
        track = df_subset[df_subset['deviceId'] == device_id]
        coords = list(zip(track['latitude'], track['longitude']))
        if len(coords) > 1:
            folium.PolyLine(
                coords, color=colors[i % len(colors)], weight=3, opacity=0.7,
                popup=f"Device: {str(device_id)[:8]}"
            ).add_to(m)
    m.save(output_html)

