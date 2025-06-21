"""
Configuration file for Vehicle Analysis Package
"""

# Directory settings
DATA_DIR = 'Data'
OUTPUT_DIR = 'Output'

# Analysis area parameters
CENTER_POINT = (22.54238, 88.304)  # Center point for analysis
ONE_MILE_RADIUS_METERS = 1600  # 1 mile ≈ 1600 meters
ONE_MILE_RADIUS_KM = 1.6  # 1 mile ≈ 1.6 km

# Analysis rectangle boundaries [lat_min, lon_min], [lat_max, lon_max]
ANALYSIS_RECTANGLE = [[22.52, 88.28], [22.56, 88.32]]

# Target area for vehicle filtering
TARGET_AREA = {
    'lat_min': 22.534,
    'lat_max': 22.546,
    'lon_min': 88.294,
    'lon_max': 88.308
}

# DBSCAN clustering parameters
DBSCAN_PARAMS = {
    'eps': 0.0003,  # Distance threshold
    'min_samples': 5  # Minimum samples per cluster
}

# Stationary detection
SPEED_THRESHOLD = 1.0  # Speed below this value (km/h) is considered stationary

# Visualization parameters
PLOT_PARAMS = {
    'figsize_default': (10, 6),
    'figsize_large': (12, 6),
    'figsize_heatmap': (16, 8),
    'color_palette': {
        'entry': '#1f77b4',  # Blue
        'exit': '#d62728'    # Red
    },
    'dwell_colors': {
        'short': 'green',    # < 5 minutes
        'medium': 'blue',    # 5-30 minutes
        'long': 'yellow',    # 30-90 minutes
        'very_long': 'orange',  # 90-300 minutes
        'extreme': 'red'     # > 300 minutes
    }
}

# File names
FILE_NAMES = {
    'input': 'cleaned_Kolkta_2_spots.csv',
    'subset': 'subset_vehicle_tracks.csv',
    'dwell_clusters': 'filtered_dwell_clusters.csv',
    'dwell_pivot': 'dwell_time_by_hour.csv',
    'entry_exit': 'entry_exit_transition_durations.csv',
    'heatmap': 'dwell_time_heatmap_per_day.png',
    'map': 'dwell_map_per_day.html'
}
