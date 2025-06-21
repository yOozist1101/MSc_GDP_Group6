"""
Simple script to run vehicle analysis
"""

from vehicle_analyzer import VehicleAnalyzer
import config


def main():
    """
    Main function to run vehicle analysis with configuration
    """
    print("=== Vehicle Trajectory Analysis ===")
    print("1. Run Complete Analysis")
    print("2. Run Preprocessing Only")
    print("3. Run Dwell Time Analysis Only")
    print("4. Run Entry/Exit Analysis Only")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nPlease select an option (1-5): ").strip()
            
            # Initialize analyzer with config parameters
            analyzer = VehicleAnalyzer(
                data_dir=config.DATA_DIR,
                output_dir=config.OUTPUT_DIR
            )
            
            # Update analyzer parameters from config
            analyzer.CENTER = config.CENTER_POINT
            analyzer.ONE_MILE_RADIUS = config.ONE_MILE_RADIUS_METERS
            analyzer.ONE_MILE_KM = config.ONE_MILE_RADIUS_KM
            analyzer.RECT_BOUNDS = config.ANALYSIS_RECTANGLE
            analyzer.eps = config.DBSCAN_PARAMS['eps']
            analyzer.min_samples = config.DBSCAN_PARAMS['min_samples']
            
            if choice == "1":
                # Run complete analysis
                print("\n--- Running Complete Analysis ---")
                results = analyzer.run_full_analysis(
                    input_file=config.FILE_NAMES['input']
                )
                
                print("\n=== Analysis Summary ===")
                print(f"Processed {len(results['subset_data'])} GPS records")
                print(f"Found {len(results['dwell_results'])} dwell clusters")
                print(f"Detected {len(results['entry_exit_results'])} entry/exit transitions")
                print(f"All outputs saved to: {config.OUTPUT_DIR}/")
                
            elif choice == "2":
                # Run preprocessing only
                print("\n--- Running Preprocessing ---")
                subset_df = analyzer.preprocess_data(
                    input_file=config.FILE_NAMES['input'],
                    output_file=config.FILE_NAMES['subset']
                )
                print(f"Preprocessing completed. {len(subset_df)} records processed.")
                
            elif choice == "3":
                # Run dwell time analysis only
                print("\n--- Running Dwell Time Analysis ---")
                try:
                    dwell_results, pivot_table = analyzer.analyze_dwell_time(
                        input_file=config.FILE_NAMES['subset'],
                        filtered_output=config.FILE_NAMES['dwell_clusters'],
                        pivot_output=config.FILE_NAMES['dwell_pivot'],
                        heatmap_output=config.FILE_NAMES['heatmap'],
                        map_output=config.FILE_NAMES['map']
                    )
                    print(f"Dwell time analysis completed. {len(dwell_results)} clusters found.")
                except FileNotFoundError:
                    print("Error: Subset file not found. Please run preprocessing first.")
                    
            elif choice == "4":
                # Run entry/exit analysis only
                print("\n--- Running Entry/Exit Analysis ---")
                try:
                    entry_exit_results = analyzer.analyze_entry_exit(
                        input_file=config.FILE_NAMES['subset'],
                        output_file=config.FILE_NAMES['entry_exit']
                    )
                    print(f"Entry/exit analysis completed. {len(entry_exit_results)} transitions detected.")
                except FileNotFoundError:
                    print("Error: Subset file not found. Please run preprocessing first.")
                    
            elif choice == "5":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice! Please enter 1, 2, 3, 4, or 5.")
                
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please check your data files and try again.")


if __name__ == "__main__":
    main()