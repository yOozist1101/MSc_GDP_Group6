import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import os
import sys
from pathlib import Path

def low_speed_threshold(row):
    """
    Determine whether the device speed is within the low speed threshold.
    Input: Vehicle speed.
    Output: A column of Boolean values. (True if vehicle speed within the 
    threshold, False otherwise)
    """
    try:
        return 0 < row['deviceSpeed'] <= 23
    except (TypeError, KeyError):
        return False 

def load_poi(poi_filepath): 
    """
    Create a BallTree for fast spatial queries.
    Input: POI file path.
    Output: poi_df Dataframe and BallTree for spatial queries.
    """
    poi_df = pd.read_csv(poi_filepath)
    poi_df = poi_df.drop(['name', 'address'], axis=1, errors='ignore')

    coords = np.array(poi_df[['lat', 'lng']].values)
    coords_rad = np.radians(coords)
    tree = BallTree(coords_rad, metric='haversine')
    
    return poi_df, tree

def match_all_pois(df, poi_df, tree, radius_km=2):
    """
    For each row in vehicle DataFrame, match the nearest POI within radius_km.
    Input: A row of DataFrame containing latitude and longitude of vehicles,
    POI DataFrame, BallTree, radius_km.
    Output: DataFrame with a new column 'poi_type' indicating the nearest POI type.
    """
    radius_rad = radius_km / 6371.0
    matched_types = []

    for _, row in df.iterrows():
        vehicle_point = np.radians([[row['latitude'], row['longitude']]])
        indices = tree.query_radius(vehicle_point, r=radius_rad)[0]

        if len(indices) > 0:
            matched_types.append(poi_df.iloc[indices[0]]['type'])
        else:
            matched_types.append('N/A')

    df['poi_type'] = matched_types
    return df

def substates(row):
    """ 
    Identify the substates of the trucks based on their states and 
    the distance to POI.
    Input: row: A row from DataFrame, including: state, low_speed, poi_type, 
    duration_secs
    Output: Substate (str): substate tags
    """
    state = row['state']
    poi_type = row.get('poi_type') 
    duration = row.get('duration_secs', 0)  # Default to 0 if not present
    transport_pois = ['ferry_terminal', 'transit_depot', 
                      'transit_station', 'railway_station']

    if state == 'on-the-move':
        if row['low_speed'] == True:
            if poi_type in transport_pois:
                return 'searching'
            else:
                return 'congestion' 
        else:
            return 'normal_driving'
    
    elif state == 'idling':
        if poi_type in transport_pois or duration > 18000 :
            return 'loading_unloading'
        else:
            return 'queuing' 

    elif state == 'stopping':
        if poi_type in transport_pois: 
                return 'loading_unloading'
        elif duration > 18000:
            if poi_type in['car_repair', 'car_wash', 
               'gas_station', 'parking', 'rest_stop', 'truck_stop',
               'hotel', 'inn', 'motel','restaurant', 'fast_food', 
               'meal_takeaway', 'convenience_store']:
                return 'resting'
            else:
                return 'loading_unloading'
        else:
            return 'resting'
        
    return 'N/A'

def substate_duration_time(df):
    """
    Calculate the duration time for each substate.
    Input: gpsTime, substate, deviceId
    Output: DataFrame with original data plus duration information
    """
    df = df.copy()
    df['gpsTime'] = pd.to_datetime(df['gpsTime'])
    df = df.sort_values(by=['deviceId', 'gpsTime']).reset_index(drop=True)
    
    # Mark each segment consecutively substate
    df['group'] = (
        df.groupby('deviceId')['substate']
        .apply(lambda x: (x != x.shift()).cumsum())
        .reset_index(drop=True)
    )
    
    # Calculate the duration of each substate segment
    segment_durations = (
        df.groupby(['deviceId', 'group'])
        .agg(
            substate=('substate', 'first'),
            start_time=('gpsTime', 'first'),
            end_time=('gpsTime', 'last')
        )
        .reset_index()
    )
    
    segment_durations['duration'] = segment_durations['end_time'] - segment_durations['start_time']
    segment_durations['substate_duration_sec'] = segment_durations['duration'].dt.total_seconds()

    # Map the duration back to the original DataFrame
    duration_map = segment_durations.set_index(['deviceId', 'group'])['substate_duration_sec'].to_dict()
    df['substate_duration_sec'] = df.apply(
        lambda row: duration_map.get((row['deviceId'], row['group']), 0),
        axis=1
    )
  
    df = df.drop('group', axis=1)
    
    return df


def process_data(df, poi_filepath):
    try:
        df['low_speed'] = df.apply(low_speed_threshold, axis=1)
        
        poi_df, poi_tree = load_poi(poi_filepath)
        
        df = match_all_pois(df, poi_df, poi_tree, radius_km=2)
        
        df['substate'] = df.apply(substates, axis=1)

        df = substate_duration_time(df)
        
        return df
    except Exception as e:
        print(f"Errors in data processing: {e}")
        raise

def get_config():
    base_dir = Path(__file__).parent
    data_dir = base_dir / "Data"
    output_dir = base_dir / "Output"

    output_dir.mkdir(exist_ok=True)
    
    cities = {
        "Mumbai": {
            "vehicle_file": data_dir / 'trucks_with_new_states_Mumbai_1.csv',
            "poi_file": data_dir / 'Mumbai_places.csv',
            "output_file": output_dir / 'processed_truck_data_Mumbai.csv'
        },
        "Kolkata": {
            "vehicle_file": data_dir / 'trucks_with_new_states_Kolkata_1.csv',
            "poi_file": data_dir / 'Kolkata_places.csv',
            "output_file": output_dir / 'processed_truck_data_Kolkata.csv'
        }
    }
    
    return cities

def validate_files(files):
    for file_type, filepath in files.items():
        if file_type != 'output_file' and not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

def main():
    cities = get_config()
    
    for city, files in cities.items():
        print(f"Start processing {city} data...")
        
        try:
            validate_files(files)

            print(f"Read {files['vehicle_file']}")
            df = pd.read_csv(files["vehicle_file"])
            print(f"{df.shape}")

            result = process_data(df, files["poi_file"])

            if 'state' in result.columns and 'substate' in result.columns:
                print(f"{city} results preview:")
                print(result[['state', 'substate']].head())
            else:
                print(f"{city} Processing complete, list: {list(result.columns)}")

            result.to_csv(files["output_file"], index=False)
            print(f"Result saved to: {files['output_file']}")
            print("-" * 50)
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue
        except Exception as e:
            print(f"Error processing {city} data: {e}")
            continue

if __name__ == '__main__':
    main()
