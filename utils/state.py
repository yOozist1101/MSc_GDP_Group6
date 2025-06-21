import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Tuple
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants definition
class VehicleStateConfig:
    """Vehicle state analysis configuration constants"""
    DEFAULT_VAR_THRESHOLD = 0.3
    DEFAULT_MIN_DURATION_SEC = 300
    DEFAULT_SPEED_THRESHOLD = 10
    DEFAULT_IGNITION_RATIO_THRESHOLD = 0.7
    
    TIME_PERIODS = {
        (2, 6): '02:00-06:00',
        (6, 10): '06:00-10:00', 
        (10, 14): '10:00-14:00',
        (14, 18): '14:00-18:00',
        (18, 22): '18:00-22:00',
        (22, 2): '22:00-02:00'  # Cross-day handling
    }
    
    STATE_STYLES = {
        'on-the-move': {'color': 'blue', 'size': 10, 'alpha': 0.5},
        'stopping': {'color': 'green', 'size': 70, 'alpha': 0.8},
        'idling': {'color': 'red', 'size': 100, 'alpha': 0.9}
    }


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate if DataFrame contains required columns
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        bool: Whether validation passes
        
    Raises:
        ValueError: Raised when required columns are missing
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
        
    return True


def assign_time_period(dt: pd.Timestamp) -> str:
    """
    Assign time period based on timestamp, supporting cross-day scenarios
    
    Args:
        dt: Timestamp
        
    Returns:
        str: Time period identifier
    """
    hour = dt.hour
    
    for (start, end), period in VehicleStateConfig.TIME_PERIODS.items():
        if start < end:  # Non-cross-day case
            if start <= hour < end:
                return period
        else:  # Cross-day case (22:00-02:00)
            if hour >= start or hour < end:
                return period
    
    # Default return (should not reach here theoretically)
    return '22:00-02:00'


def label_states_updated(df: pd.DataFrame, 
                        var_threshold: float = VehicleStateConfig.DEFAULT_VAR_THRESHOLD,
                        min_duration_sec: int = VehicleStateConfig.DEFAULT_MIN_DURATION_SEC,
                        speed_threshold: float = VehicleStateConfig.DEFAULT_SPEED_THRESHOLD) -> pd.DataFrame:
    """
    Label states for a single vehicle trajectory
    
    Args:
        df: Vehicle GPS data DataFrame
        var_threshold: Speed variance threshold for distinguishing idling from stopping
        min_duration_sec: Minimum duration in seconds, stops below this won't be labeled
        speed_threshold: Speed threshold, below this is considered non-moving state
        
    Returns:
        pd.DataFrame: DataFrame with state labels
        
    Raises:
        ValueError: Raised when input data doesn't meet requirements
    """
    # Input validation
    required_columns = ['gpsTime', 'deviceSpeed', 'ignition']
    validate_dataframe(df, required_columns)
    
    try:
        df = df.sort_values(by='gpsTime').copy()
        df['gpsTime'] = pd.to_datetime(df['gpsTime'])
        
        # Mark non-moving states
        df['non_move'] = df['deviceSpeed'] < speed_threshold
        df['state_group'] = (df['non_move'] != df['non_move'].shift()).cumsum()

        # Calculate statistics for each stop segment
        segment_stats = (
            df[df['non_move']]
            .groupby(['state_group'])
            .agg(
                start_time=('gpsTime', 'min'),
                end_time=('gpsTime', 'max'),
                duration=('gpsTime', lambda x: (x.max() - x.min()).total_seconds()),
                speed_var=('deviceSpeed', 'var'),
                ignition_ratio=('ignition', 'mean')
            )
            .reset_index()
        )

        # Filter segments meeting minimum duration requirement
        idling_segments = segment_stats[segment_stats['duration'] >= min_duration_sec].copy()

        def get_idling_type(row: pd.Series) -> str:
            """Determine stop type based on speed variance and ignition ratio"""
            if (row['speed_var'] > var_threshold or 
                row['ignition_ratio'] > VehicleStateConfig.DEFAULT_IGNITION_RATIO_THRESHOLD):
                return 'idling'
            else:
                return 'stopping'

        # Assign states to qualified stop segments
        idling_segments['state'] = idling_segments.apply(get_idling_type, axis=1)
        state_map = idling_segments.set_index('state_group')['state'].to_dict()
        duration_map = segment_stats.set_index('state_group')['duration'].to_dict()

        # Assign states to all data points
        df['state'] = 'on-the-move'
        mask = df['non_move']
        df.loc[mask, 'state'] = df[mask].apply(
            lambda row: state_map.get(row['state_group'], 'on-the-move'),
            axis=1
        )

        df['duration_secs'] = df['state_group'].map(duration_map)
        
        # Assign time periods
        df['time_period'] = df['gpsTime'].apply(assign_time_period)
        
        logger.info(f"Successfully processed vehicle trajectory data with {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error occurred while processing vehicle trajectory data: {e}")
        raise


def process_all_devices(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Batch process all vehicle trajectory data
    
    Args:
        df_all: DataFrame containing all vehicle data
        
    Returns:
        pd.DataFrame: Processed and merged data
    """
    required_columns = ['deviceId', 'gpsTime', 'deviceSpeed', 'ignition']
    validate_dataframe(df_all, required_columns)
    
    result_list = []
    failed_devices = []
    
    total_devices = df_all['deviceId'].nunique()
    logger.info(f"Starting to process data for {total_devices} vehicles")
    
    for i, (device_id, group) in enumerate(df_all.groupby('deviceId')):
        try:
            processed = label_states_updated(group)
            result_list.append(processed)
            
            if (i + 1) % 100 == 0:  # Output progress every 100 devices
                logger.info(f"Processed {i + 1}/{total_devices} devices")
                
        except Exception as e:
            logger.warning(f"Error processing device {device_id}: {e}")
            failed_devices.append(device_id)
            continue
    
    if failed_devices:
        logger.warning(f"Number of failed devices: {len(failed_devices)}")
        logger.debug(f"Failed device IDs: {failed_devices}")
    
    if not result_list:
        raise ValueError("No device data was successfully processed")
        
    result_df = pd.concat(result_list, ignore_index=True)
    logger.info(f"Processing completed successfully with {len(result_df)} records")
    
    return result_df


def visualize_truck_states_by_index(df_with_states: pd.DataFrame, 
                                  index: int = 0,
                                  figsize: Tuple[int, int] = (12, 8),
                                  save_path: Optional[str] = None) -> None:
    """
    Visualize state trajectory for vehicle at specified index
    
    Args:
        df_with_states: Dataset containing state fields
        index: Position in sorted device list (0 means first vehicle)
        figsize: Figure size
        save_path: Save path, saves image if provided
        
    Raises:
        ValueError: Raised when index is out of range
    """
    required_columns = ['deviceId', 'longitude', 'latitude', 'state']
    validate_dataframe(df_with_states, required_columns)
    
    # Get sorted device list
    sorted_devices = sorted(df_with_states['deviceId'].unique())
    
    if index >= len(sorted_devices) or index < 0:
        raise ValueError(f"Index {index} out of range (0-{len(sorted_devices)-1})")
    
    device_id = sorted_devices[index]
    df_device = df_with_states[df_with_states['deviceId'] == device_id]
    
    if df_device.empty:
        logger.warning(f"Device {device_id} has no data")
        return

    # Plot state diagram
    plt.figure(figsize=figsize)
    
    for state, style in VehicleStateConfig.STATE_STYLES.items():
        subset = df_device[df_device['state'] == state]
        if not subset.empty:
            plt.scatter(
                subset['longitude'], subset['latitude'],
                s=style['size'], c=style['color'], 
                alpha=style['alpha'], label=f"{state} ({len(subset)})"
            )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Vehicle Trajectory State Visualization - Device {device_id} (Index {index})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Image saved to: {save_path}")
    
    plt.show()


def calculate_state_speed_stats(df: pd.DataFrame, 
                              state_col: str = 'state', 
                              speed_col: str = 'deviceSpeed') -> pd.DataFrame:
    """
    Calculate speed statistics for each state
    
    Args:
        df: DataFrame containing state and speed data
        state_col: State column name
        speed_col: Speed column name
        
    Returns:
        pd.DataFrame: Speed statistics for each state
    """
    validate_dataframe(df, [state_col, speed_col])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # Ignore warnings for empty data
        
        state_speed_stats = (
            df
            .groupby(state_col)[speed_col]
            .agg(['mean', 'var', 'std', 'count'])
            .rename(columns={
                'mean': 'speed_mean', 
                'var': 'speed_variance',
                'std': 'speed_std',
                'count': 'record_count'
            })
            .round(2)
            .reset_index()
        )
    
    logger.info("Speed statistics calculation completed")
    return state_speed_stats


def calculate_state_duration_stats(df: pd.DataFrame, 
                                 state_col: str = 'state', 
                                 duration_col: str = 'duration_secs') -> pd.DataFrame:
    """
    Calculate duration statistics for each state
    
    Args:
        df: DataFrame containing state and duration data
        state_col: State column name
        duration_col: Duration column name (in seconds)
        
    Returns:
        pd.DataFrame: Duration statistics for each state
    """
    validate_dataframe(df, [state_col, duration_col])
    
    # Only analyze records with duration information
    df_with_duration = df.dropna(subset=[duration_col])
    
    if df_with_duration.empty:
        logger.warning("No duration data found")
        return pd.DataFrame()
    
    state_duration_stats = (
        df_with_duration
        .groupby(state_col)[duration_col]
        .agg(['mean', 'median', 'std', 'count'])
        .rename(columns={
            'mean': 'avg_duration_sec', 
            'median': 'median_duration_sec',
            'std': 'std_duration_sec',
            'count': 'segment_count'
        })
        .round(2)
        .reset_index()
    )
    
    # Add average duration in hours
    state_duration_stats['avg_duration_hours'] = (
        state_duration_stats['avg_duration_sec'] / 3600
    ).round(2)
    
    logger.info("Duration statistics calculation completed")
    return state_duration_stats


def main():
    """Main function - demonstrates complete data processing workflow"""
    try:
        # Read data
        input_file = 'cleaned_gps_data_Mumbai.csv'  # Configurable
        logger.info(f"Starting to read data file: {input_file}")
        
        df = pd.read_csv(input_file)
        logger.info(f"Successfully read data with {len(df)} records and {df['deviceId'].nunique()} devices")
        
        # Process all device data
        df_with_states = process_all_devices(df)
        
        # Clean unnecessary columns
        columns_to_drop = ['non_move', 'state_group']
        # Only drop existing columns
        existing_columns_to_drop = [col for col in columns_to_drop if col in df_with_states.columns]
        if 'makerDescription' in df_with_states.columns:
            existing_columns_to_drop.append('makerDescription')
        if 'modelDescription' in df_with_states.columns:
            existing_columns_to_drop.append('modelDescription')
            
        if existing_columns_to_drop:
            df_with_states = df_with_states.drop(columns=existing_columns_to_drop)

        # Calculate statistics
        logger.info("Calculating state statistics...")
        state_speed_stats = calculate_state_speed_stats(df_with_states)
        print("\n=== Speed Statistics by State ===")
        print(state_speed_stats.to_string(index=False))

        state_duration_stats = calculate_state_duration_stats(df_with_states)
        if not state_duration_stats.empty:
            print("\n=== Duration Statistics by State ===")
            print(state_duration_stats.to_string(index=False))

        # Visualize first device
        logger.info("Generating visualization...")
        visualize_truck_states_by_index(df_with_states, index=0)# e.g. The 0th vehicle after visualization sorting

        # Save results
        output_path = "trucks_with_new_states_Mumbai.csv"  # Configurable
        df_with_states.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")
        
        # Output processing summary
        print(f"\n=== Processing Summary ===")
        print(f"Total records: {len(df_with_states)}")
        print(f"Number of devices: {df_with_states['deviceId'].nunique()}")
        print("State distribution:")
        print(df_with_states['state'].value_counts().to_string())

    except Exception as e:
        logger.error(f"Main program execution failed: {e}")
        raise



if __name__ == "__main__":
    main()
