import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple

from utils.data_preprocessing import *
from utils.state import *
from utils.substate import *
from utils.evaluation import *
from utils.prediction_model import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MainPipeline:
    """
    Main pipeline class for analysis
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the pipeline
        
        Args:
            base_dir: Base directory path (defaults to current directory)
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.data_dir = self.base_dir / "Data"
        self.output_dir = self.base_dir / "Output"
        self.models_dir = self.output_dir / "Models"
        
        # Create directories
        for dir_path in [self.data_dir, self.output_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Pipeline configuration: could be costumized according to the chosen pilot
        self.config = {
            'poi_radius_meters': 5000,
            'banned_hours': {
                'single_window': (time(6, 0), time(22, 0)),
                'peak_hours': [(time(8, 0), time(11, 0)), (time(17, 0), time(21, 0))]
            }
        }

        # Initialize analyzer
        self.analyzer = TrafficAnalysisSystem(
            input_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        
        logger.info(f"Pipeline initialized with base directory: {self.base_dir}")

    def dp_sample(self, city: str = None):
        """
        Data preprocessing and cleaning
        
        Args:
            city: City name for processing (if None, processes all available data)
            
        Returns:
            Dict containing output file paths
        """
        logger.info("STEP 1: DATA PREPROCESSING AND CLEANING")

        results = {}

        try:
            day_pattern = "sample_data_day_*.csv"
            day_files = list(self.data_dir.glob(day_pattern))
            
            if not day_files:
                logger.warning(f"No files found matching pattern: {day_pattern}")
                return results
            
            logger.info(f"Found {len(day_files)} day files")
            
            # Concatenate daily files
            merged_file = self.output_dir / "merged_sample_data.csv"
            concat_files(
                file_list=[str(f) for f in sorted(day_files)],
                output_file=str(merged_file),
                chunksize=100000
            )
            results['merged_file'] = str(merged_file)
            logger.info(f"Merged file created: {merged_file}")
            
            # Match trajectories to POI
            poi_file = self.data_dir / "Sample_Target-point.csv"
            current_file = merged_file
            
            if poi_file.exists():
                matched_file = self.output_dir / "matched_sample_trajectories.csv"
                match_trajectories_to_poi(
                    trajectory_file=str(current_file),
                    poi_file=str(poi_file),
                    output_file=str(matched_file),
                    radius_meters=self.config['poi_radius_meters']
                )
                results['matched_file'] = str(matched_file)
                current_file = matched_file
                logger.info(f"POI matching completed: {matched_file}")
            else:
                logger.info("No POI file found, skipping POI matching")
                results['matched_file'] = str(current_file)
                       
            # Clean GPS data
            cleaned_file = self.output_dir / f"cleaned_gps_data_{'all' if not city else city}.csv"
            process_and_clean_gps(
                input_file=results['matched_file'],
                output_file=str(cleaned_file)
            )
            results['cleaned_file'] = str(cleaned_file)
            logger.info(f"Cleaned GPS data: {cleaned_file}")
            
            # Create visualization
            if poi_file.exists():
                viz_file = self.output_dir / "sample_trajectories_map.html"
                try:
                    visualize_trajectories(
                        trajectory_file=str(cleaned_file),
                        poi_file=str(poi_file),
                        output_html=str(viz_file),
                        max_vehicles=300
                    )
                    results['visualization'] = str(viz_file)
                    logger.info(f"Visualization created: {viz_file}")
                except Exception as e:
                    logger.warning(f"Visualization failed: {e}")
            
            logger.info("Sample data preprocessing completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in sample data preprocessing: {e}")
            raise

    def dp_city(self, city: str, day_range: range = range(1, 4)):
        """
        Processing of city-specific data
        
        Args.
            city: City name
            day_range: range of days to process
            
        Returns: Dict: Dictionary containing the path to the output file.
            Dict: Dictionary containing the path to the output file
        """
        logger.info(f"STEP 1: CITY DATA PREPROCESSING - {city}")
        
        results = {}
        
        try:
            # Search a city day file
            day_files = []
            for i in day_range:
                day_file = self.data_dir / f"{city}_Day_{i:02d}.csv"
                if day_file.exists():
                    day_files.append(day_file)
            
            if not day_files:
                logger.warning(f"No day files found for city: {city}")
                return results
            
            logger.info(f"Found {len(day_files)} day files for {city}")
            
            # Merge file
            merged_file = self.output_dir / f"merged_{city}_data.csv"
            concat_files(
                file_list=[str(f) for f in day_files],
                output_file=str(merged_file),
                chunksize=100000
            )
            results['merged_file'] = str(merged_file)
            logger.info(f"Merged file created: {merged_file}")
            
            # Match POI
            poi_file = self.data_dir / f"{city}_places.csv"
            current_file = merged_file
            
            if poi_file.exists():
                matched_file = self.output_dir / f"matched_{city}_trajectories.csv"
                match_trajectories_to_poi(
                    trajectory_file=str(current_file),
                    poi_file=str(poi_file),
                    output_file=str(matched_file),
                    radius_meters=self.config['poi_radius_meters']
                )
                results['matched_file'] = str(matched_file)
                current_file = matched_file
                logger.info(f"POI matching completed: {matched_file}")
            else:
                logger.warning(f"POI file not found: {poi_file}")
                results['matched_file'] = str(current_file)
            
            # Clean GPS data
            cleaned_file = self.output_dir / f"cleaned_{city}_data.csv"
            process_and_clean_gps(
                input_file=str(current_file),
                output_file=str(cleaned_file)
            )
            results['cleaned_file'] = str(cleaned_file)
            logger.info(f"GPS data cleaned: {cleaned_file}")
            
            # Create visualization
            if poi_file.exists():
                viz_file = self.output_dir / f"{city}_trajectories_map.html"
                try:
                    visualize_trajectories(
                        trajectory_file=str(cleaned_file),
                        poi_file=str(poi_file),
                        output_html=str(viz_file),
                        max_vehicles=300
                    )
                    results['visualization'] = str(viz_file)
                    logger.info(f"Visualization created: {viz_file}")
                except Exception as e:
                    logger.warning(f"Visualization failed: {e}")
            
            logger.info(f"City data preprocessing completed for {city}")
            return results
            
        except Exception as e:
            logger.error(f"Error in city data preprocessing: {e}")
            raise

    def dp(self, mode="sample", city=None, day_range=range(1, 4)):
        """
        Unified data preprocessing portal
        
        Args.
            mode: processing mode (“sample” or “city”)
            city: city name (required if mode=“city”)
            day_range: range of days
            
        Returns: city name (required for mode=“city”)
            Dict: the result of the process
        """
        if mode == "sample":
            return self.dp_sample()
        elif mode == "city":
            if not city:
                raise ValueError("City name is required for city mode")
            return self.dp_city(city, day_range)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'sample' or 'city'")

        
    def state(self, input_file: str, city: str = "all"):
        """
        Step 2: State analysis (idling, on-the-move, stopping)
        
        Args:
            input_file: Input cleaned GPS data file
            city: City name for output naming
            
        Returns:
            Output file path with state information
        """

        logger.info("STEP 2: STATE CLASSIFICATION")

        try:
            # Load cleaned data
            df = pd.read_csv(input_file)

            # Process truck states
            df_state = process_all_devices(df)

            # Calculate & Display statistics
            state_v = calculate_state_speed_stats(df_state)
            state_dur_t = calculate_state_duration_stats(df_state)
            print("\nSpeed Statistics by State:", state_v.to_string(index=False))
            print("\nDuration Statistics by State:", state_dur_t.to_string(index=False))
            print("\nState Distribution:", df_state['state'].value_counts())

            # Generate sample visulization
            try:
                logger.info("Sample visulization")
                viz_path = self.output_dir / f"vehicle_states_visualization_{city}.png"
                visualize_truck_states_by_index(
                    df_state, 
                    index=0, 
                    save_path=str(viz_path)
                )
            except Exception as e:
                logger.warning(f"Visualization generation failed: {e}")

            # Output results and statistics
            output_file = self.output_dir/ f"trucks_with_states_{city}.csv"
            df_state.to_csv(output_file, index = False)

            logger.info(f"Step 2 results saved to: {output_file}")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error in Step 2: {e}")
            raise

    def substate(self, input_file: str, city: str):
        """
        Step 3: Substate classification
        
        Args:
            input_file: Input file with state information
            city: City name for POI matching
            
        Returns:
            Output file path with substate information
        """
        logger.info("STEP 3: SUBSTATE CLASSIFICATION")

        try:
            # Load data and POI file
            df = pd.read_csv(input_file)

            poi = self.data_dir / f"{city}_POI.csv"
            if not poi.exists():
                    poi = self.data_dir / "sample_POI.csv"
                    return input_file
            
            # Substate classification and brief visulization
            df_substate = process_data(df, str(poi))
            print(df_substate['substate'].value_counts())

            # Save results
            output_file = self.output_dir / f"trucks_substates_{city}.csv"
            df_substate.to_csv(output_file, index = False)

            logger.info(f"Step 3 results saved to: {output_file}")

            return str(output_file)

        except Exception as e:
            logger.error(f"Error in Step 3: {e}")
            raise     

    def evaluation(self, input_file: str, city: str):
        """
        Step 4: Evaluation and visualization of classification results
    
        Args:
            input_file: Input file with complete analysis
            city: City name for analysis
        """
        
        logger.info("STEP 4: EVALUATION")
    
        try:
            # Load final data
            logger.info(f"Loading complete data from: {input_file}")
            df = pd.read_csv(input_file)
            
            # Fix datetime column
            df = self.fix_datetime_column(df, 'gpsTime')
            
            # Basic substate/state analysis
            logger.info("Performing comprehensive substate analysis")
            try:
                self.analyzer.analyze_and_save_results(df, f'{city}_comprehensive')
            except Exception as e:
                logger.warning(f"Basic analysis failed: {e}")
        
            # Vehicle class analysis
            if 'vehicleClass' in df.columns:
                logger.info("Analyzing vehicle classes")
                try:
                    self.analyzer.analyze_vehicle_classes(df, f'vehicle_classes_{city}.csv')
                except Exception as e:
                    logger.warning(f"Vehicle class analysis failed: {e}")
            else:
                logger.info("No vehicle class data found, skipping vehicle class analysis")
        
            # Temporal analysis - check if sufficient data for weekday/weekend analysis
            temporal_analysis_possible = self.check_temporal_analysis_feasible(df)
            
            if temporal_analysis_possible:
                logger.info("Performing temporal analysis")
                try:
                    self.analyzer.plot_weekday_comparison(df, output_file=f'weekday_comparison_{city}.png')
                    self.analyzer.plot_normalized_comparison(df, output_file=f'normalized_comparison_{city}.png')
                    logger.info("Temporal analysis completed successfully")
                except Exception as e:
                    logger.warning(f"Temporal analysis failed: {e}")
            else:
                logger.info("Skipping temporal analysis - insufficient weekday/weekend data or missing required columns")
        
            # Traffic violation analysis
            logger.info("Analyzing traffic violations")
            violation_analysis_possible = self.check_violation_analysis_feasible(df)
            
            if violation_analysis_possible:
                try:
                    # Single window violations
                    self.analyzer.calculate_violation_ratio(
                        df,
                        banned_hours=self.config['banned_hours']['single_window'],
                        output_file=f'violations_single_{city}.png'
                    )
                
                    # Peak hour violations
                    self.analyzer.calculate_mumbai_violations(
                        df,
                        banned_windows=self.config['banned_hours']['peak_hours'],
                        output_file=f'violations_peak_{city}.png'
                    )
                    logger.info("Violation analysis completed successfully")
                except Exception as e:
                    logger.warning(f"Violation analysis failed: {e}")
            else:
                logger.info("Skipping violation analysis - missing required columns or insufficient data")
        
            # Direction analysis (if target data available)
            direction_analysis_possible = self.check_direction_analysis_feasible(df)
            
            if direction_analysis_possible:
                logger.info("Performing direction analysis")
                try:
                    target_file = self.data_dir / "Sample_Target-point.csv"
                    targets_df = pd.read_csv(target_file)
                    direction_df = self.analyzer.classify_trajectory_directions(df, targets_df)
                
                    if direction_df is not None and len(direction_df) > 0:
                        merged_df = self.analyzer.merge_direction_info(
                            df, direction_df, f'trajectory_with_direction_{city}.csv'
                        )
                    
                        if merged_df is not None and len(merged_df) > 0:
                            self.analyzer.analyze_direction_distribution(
                                merged_df, output_prefix=f'direction_{city}'
                            )
                            # Only perform timing analysis if datetime format is correct
                            if pd.api.types.is_datetime64_any_dtype(merged_df['gpsTime']):
                                self.analyzer.analyze_direction_timing(
                                    merged_df, output_prefix=f'direction_timing_{city}'
                                )
                    logger.info("Direction analysis completed successfully")
                except Exception as e:
                    logger.warning(f"Direction analysis failed: {e}")
            else:
                logger.info("Skipping direction analysis - missing target file, Target column, or insufficient data")
        
            # Create traffic maps
            map_analysis_possible = self.check_map_analysis_feasible(df)
            
            if map_analysis_possible:
                logger.info("Creating traffic maps")
                try:
                    top_substates = df['substate'].value_counts().head(3).index.tolist()
                    map_created = False
                    for substate in top_substates:
                        substate_data = df[df['substate'] == substate]
                        if len(substate_data) > 100:
                            try:
                                self.analyzer.create_traffic_map(
                                    df, substate, f'{city}_{substate}_map.html'
                                )
                                map_created = True
                            except Exception as e:
                                logger.warning(f"Map creation failed for {substate}: {e}")
                    
                    if map_created:
                        logger.info("Traffic maps created successfully")
                    else:
                        logger.info("No traffic maps created - insufficient data for individual substates")
                except Exception as e:
                    logger.warning(f"Traffic map creation failed: {e}")
            else:
                logger.info("Skipping traffic maps - missing required columns or insufficient data")
        
            logger.info("Step 4 completed")
        
        except Exception as e:
            logger.error(f"Error in Step 4: {e}")
            raise

    def check_temporal_analysis_feasible(self, df):
        """
        Check if temporal analysis is feasible
        
        Returns:
            bool: True if temporal analysis is feasible, False otherwise
        """
        try:
            # Check required columns
            required_columns = ['gpsTime', 'substate']
            if not all(col in df.columns for col in required_columns):
                logger.info(f"Missing required columns for temporal analysis: {[col for col in required_columns if col not in df.columns]}")
                return False
            
            # Check data volume
            if len(df) < 10:
                logger.info("Insufficient data points for temporal analysis")
                return False
            
            # Check time format
            if not pd.api.types.is_datetime64_any_dtype(df['gpsTime']):
                logger.info("gpsTime is not in datetime format")
                return False
            
            # Check if there is sufficient time span (at least different dates)
            df_temp = df.copy()
            df_temp['date'] = df_temp['gpsTime'].dt.date
            unique_dates = df_temp['date'].nunique()
            
            if unique_dates < 2:
                logger.info(f"Insufficient time span for temporal analysis (only {unique_dates} unique date(s))")
                return False
            
            # Check if there is both weekday and weekend data
            df_temp['day_of_week'] = df_temp['gpsTime'].dt.dayofweek
            weekdays = df_temp[df_temp['day_of_week'] < 5]
            weekends = df_temp[df_temp['day_of_week'] >= 5]
            
            if len(weekdays) == 0:
                logger.info("No weekday data found")
                return False
            
            if len(weekends) == 0:
                logger.info("No weekend data found")
                return False
            
            # Check if there is sufficient substate diversity
            substate_counts = df['substate'].value_counts()
            if len(substate_counts) < 2:
                logger.info("Insufficient substate variety for temporal analysis")
                return False
            
            logger.info("Temporal analysis is feasible")
            return True
            
        except Exception as e:
            logger.warning(f"Error checking temporal analysis feasibility: {e}")
            return False

    def check_violation_analysis_feasible(self, df):
        """
        Check if violation analysis is feasible
        """
        try:
            required_columns = ['gpsTime', 'latitude', 'longitude']
            if not all(col in df.columns for col in required_columns):
                return False
            
            if not pd.api.types.is_datetime64_any_dtype(df['gpsTime']):
                return False
            
            if len(df) < 10:
                return False
            
            # Check if location data is valid
            if df['latitude'].isna().all() or df['longitude'].isna().all():
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking violation analysis feasibility: {e}")
            return False

    def check_direction_analysis_feasible(self, df):
        """
        Check if direction analysis is feasible
        """
        try:
            target_file = self.data_dir / "Sample_Target-point.csv"
            
            if not target_file.exists():
                return False
            
            required_columns = ['Target', 'latitude', 'longitude']
            if not all(col in df.columns for col in required_columns):
                return False
            
            if len(df) < 10:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking direction analysis feasibility: {e}")
            return False

    def check_map_analysis_feasible(self, df):
        """
        Check if map analysis is feasible
        """
        try:
            required_columns = ['latitude', 'longitude', 'substate']
            if not all(col in df.columns for col in required_columns):
                return False
            
            if len(df) < 50:  # Maps need more data points
                return False
            
            # Check if location data is valid
            if df['latitude'].isna().all() or df['longitude'].isna().all():
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking map analysis feasibility: {e}")
            return False

    def fix_datetime_column(self, df, column_name='gpsTime'):
        """
        Fix datetime column in DataFrame
        
        Args:
            df: DataFrame to fix
            column_name: Name of the time column to fix
            
        Returns:
            Fixed DataFrame
        """
        if column_name not in df.columns:
            logger.warning(f"Column {column_name} not found in DataFrame")
            return df
        
        df = df.copy()
        
        try:
            # If already datetime type, return directly
            if pd.api.types.is_datetime64_any_dtype(df[column_name]):
                logger.info(f"{column_name} is already in datetime format")
                return df
            
            logger.info(f"Converting {column_name} to datetime format")
            
            # Try different conversion methods
            conversion_methods = [
                lambda x: pd.to_datetime(x),  # Direct conversion
                lambda x: pd.to_datetime(x, unit='ms'),  # Millisecond timestamp
                lambda x: pd.to_datetime(x, unit='s'),   # Second timestamp
                lambda x: pd.to_datetime(x, infer_datetime_format=True),  # String parsing
            ]
            
            for i, method in enumerate(conversion_methods):
                try:
                    df[column_name] = method(df[column_name])
                    logger.info(f"Successfully converted using method {i+1}")
                    return df
                except:
                    continue
            
            # If all methods fail, create dummy timestamps
            logger.warning(f"Could not convert {column_name}, creating dummy timestamps")
            df[column_name] = pd.date_range(
                start='2023-01-01', 
                periods=len(df), 
                freq='1min'
            )
            
        except Exception as e:
            logger.error(f"Error fixing datetime column: {e}")
            df[column_name] = pd.date_range(
                start='2023-01-01', 
                periods=len(df), 
                freq='1min'
            )
        
        return df
    
    def pm(self, input_file: str, city: str):
        """
        Step 5: Train and evaluate predictive models
        
        Args:
            input_file: Input file with complete analysis
            city: City name for model naming
        """
        logger.info("STEP 5: PREDICTIVE MODELING")

        try:
            # Load data
            df = pd.read_csv(input_file)

            # Check columns
            required_col = ['deviceSpeed', 'orientation', 'latitude', 'longitude', 'ignition', 'state']
            missing_cols = [col for col in required_col if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns for modeling: {missing_cols}")
                return
        
            # Initialize classifier
            classifier = VehicleStateClassifier()

            # Train and Evaluate model
            logger.info("Training predictive model")
            train = classifier.fit(df, epochs = 20, batch_size = 32, verbose = 1)
            logger.info("Evaluating model performance")
            classifier.evaluate(plot_confusion_matrix=True)

            # Save model
            model_dir = self.output_dir / "Models"
            model_dir.mkdir(exist_ok=True)

            classifier.save_model(
                model_path=str(model_dir / f"state_classifier_{city}.keras"),
                scaler_path=str(model_dir / f"features_scaler_{city}.pkl"),
                encoder_path=str(model_dir / f"labels_encoder_{city}.pkl")
            )
            logger.info(f"Step 5 completed")
            
        except Exception as e:
            logger.error(f"Error in Step 5: {e}")
            raise

    def runpipeline(self, city: str = None):
        """
        Run the full state analysis pipeline
        
        Args:
            city: City to process (if None, processes general data)
        """
        logger.info("START ANALYSIS PIPELINE")
        
        try:
            # Step 1: Data Preprocessing
            preprocessing_results = self.dp(city)
            
            if not preprocessing_results.get('cleaned_file'):
                logger.error("No cleaned data file produced, stopping pipeline")
                return
            
            # Step 2: State Analysis
            state_file = self.state(
                preprocessing_results['cleaned_file'], 
                city or 'all'
            )
            
            # Step 3: Substate Analysis
            substate_file = self.substate(state_file, city or 'all')
            final_file = substate_file
            
            # Step 4: Evaluation
            self.evaluation(final_file, city or 'all')
            
            # Step 5: Predictive Modeling
            self.pm(state_file, city or 'all')
            
            # Pipeline completion
            logger.info("PIPELINE COMPLETED")
            logger.info(f"Final output directory: {self.output_dir}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """
    Main entry point for the pipeline
    """
    print("State Analysis Pipeline")
    
    try:
        # Initialize pipeline
        pipeline = MainPipeline()
        
        # Simplified menu
        print("1. Run complete pipeline")
        print("2. Exit")
        
        choice = input("\nSelect option (1-2): ").strip()
        
        if choice == "1":
            # Run complete pipeline
            mode_choice = input("Run for (1) sample data or (2) city data? ").strip()
            if mode_choice == "1":
                print("Running complete pipeline for sample data...")
                
                # Step 1: Data preprocessing
                results = pipeline.dp(mode="sample")
                print("Step 1 completed")
                
                if 'cleaned_file' in results:
                    # Step 2: State classification
                    state_file = pipeline.state(results['cleaned_file'], 'sample')
                    print("Step 2 completed")
                    
                    # Step 3: Substate classification - USE OUTPUT FROM STEP 2
                    substate_file = pipeline.substate(state_file, 'sample')  # Changed this line
                    print("Step 3 completed")
                    
                    # Step 4: Evaluation - USE OUTPUT FROM STEP 3
                    pipeline.evaluation(substate_file, 'sample')  # Changed this line
                    print("Step 4 completed")
                    
                    # Step 5: Model training - USE OUTPUT FROM STEP 2 (states only)
                    pipeline.pm(state_file, 'sample')  # Changed this line
                    print("Step 5 completed")
                    
                    print("Complete pipeline finished")
                
            elif mode_choice == "2":
                city = input("Enter city name: ").strip()
                print(f"Running complete pipeline for {city}...")
                
                # Step 1: Data preprocessing
                results = pipeline.dp(mode="city", city=city)
                print("Step 1 completed")
                
                if 'cleaned_file' in results:
                    # Step 2: State classification
                    state_file = pipeline.state(results['cleaned_file'], city)
                    print("Step 2 completed")
                    
                    # Step 3: Substate classification - USE OUTPUT FROM STEP 2
                    substate_file = pipeline.substate(state_file, city)  # Changed this line
                    print("Step 3 completed")
                    
                    # Step 4: Evaluation - USE OUTPUT FROM STEP 3
                    pipeline.evaluation(substate_file, city)  # Changed this line
                    print("Step 4 completed")
                    
                    # Step 5: Model training - USE OUTPUT FROM STEP 2 (states only)
                    pipeline.pm(state_file, city)  # Changed this line
                    print("Step 5 completed")
                    
                    print(f"Complete pipeline finished for {city}!")
            else:
                print("Invalid mode choice")
        
        elif choice == "2":
            print("Goodbye!")
        
        else:
            print("Invalid choice. Please select 1-2.")
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        logger.error(f"Main program error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()