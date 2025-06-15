import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

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


def process_data(df, poi_filepath):
    df['low_speed'] = df.apply(low_speed_threshold, axis=1)
    
    poi_df, poi_tree = load_poi(poi_filepath)
    
    df = match_all_pois(df, poi_df, poi_tree, radius_km=2)
    
    df['substate'] = df.apply(substates, axis=1)
    
    return df  


if __name__ == '__main__':
    cities = {
        "Mumbai": {
            "vehicle_file": 'C:/Users/18611/Desktop/trucks_with_new_states_Mumbai.csv',
            "poi_file": 'C:/Users/18611/Desktop/Mumbai_places.csv',
            "output_file": 'C:/Users/18611/Desktop/processed_truck_data_Mumbai.csv'
        },
        "Kolkata": {
            "vehicle_file": 'C:/Users/18611/Desktop/trucks_with_new_states_Kolkata.csv',
            "poi_file": 'C:/Users/18611/Desktop/Kolkata_places.csv',
            "output_file": 'C:/Users/18611/Desktop/processed_truck_data_Kolkata.csv'
        }
    }


    for city, files in cities.items():
        df = pd.read_csv(files["vehicle_file"])
        result = process_data(df, files["poi_file"])
        print(f"{city} preview:\n", result[['state', 'substate']].head())
        result.to_csv(files["output_file"], index=False)
