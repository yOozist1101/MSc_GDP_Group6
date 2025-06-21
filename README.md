<div align="center">
<h1>Big Data and AI logistics model for First-Last Mile Issues in India </h1>
</div>

## Table of Contents
- [Background](#Background)
- [Data Preprocessing](#data-preprocessing)
- [Methods](#methods)
   - [State Classification](#state-classification)
   - [Substate Classification](#substate-classification)
   - [Prediction Model](#prediction-model)
- [Case Study](#case-study)

## Background

## Data Preprocessing

## Methods

### State Classification
## · Features
- **State Labeling**: Classifies GPS points into:
  - `on-the-move`: regular movement
  - `stopping`: short-duration halts with low ignition ratio and low speed variance
  - `idling`: long-duration stops with high ignition ratio or speed variance
- **Segment Analysis**: Detects continuous stop segments based on speed threshold and temporal gaps
- **Time Period Assignment**: Categorizes data into time blocks (e.g., 02:00–06:00) for temporal analysis
- **Batch Processing**: Handles trajectories of multiple devices efficiently with error handling
- **Visualization**: Plots trajectory points by state on a scatter map
- **Statistics**: Calculates speed and stop duration summaries for each state

## · Folder Structure

```

├── state_readme.py                     # Main utility script
├── cleaned_gps_data_Mumbai.csv         # Example input file (optional)
└── trucks_with_new_states_Mumbai.csv  # Output file after processing
```


## · Usage

### · Functions Overview

- `process_all_devices(df)`: batch processing for each vehicle  
- `label_states_updated(df)`: segment-wise state labeling logic  
- `calculate_state_speed_stats(df)`: prints mean and std of speeds by state  
- `calculate_state_duration_stats(df)`: prints average stop duration by state  
- `visualize_truck_states_by_index(df, index=0)`: visualize one truck's trajectory using matplotlib  

---

### · How to Use the Functions

After importing the script or copying the functions into your environment, follow these usage examples:

```python
import pandas as pd
from state_readme import *

# Step 1: Load your GPS data
df = pd.read_csv("cleaned_gps_data_Mumbai.csv")

# Step 2: Apply state labeling
df_labeled = process_all_devices(df)

# Step 3: (Optional) Print statistics
calculate_state_speed_stats(df_labeled)
calculate_state_duration_stats(df_labeled)

# Step 4: (Optional) Visualize a sample vehicle
visualize_truck_states_by_index(df_labeled, index=0)
```

## · Future Extensions

- Heatmap visualization using `folium`  
- Streamlit dashboard for interactive map and filters  
- Add clustering or anomaly detection for idling patterns  
- Integrate warehouse/port POI data for spatial context  

> · **Notes**
>
> - Input timestamps must be convertible to `datetime` using `pd.to_datetime`
> - Devices with invalid timestamps will be skipped and logged
> - Visualization uses `matplotlib` and is intended for Jupyter or script environments

### Substate Classification

### Prediction Model

## Case Study

