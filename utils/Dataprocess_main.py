from gps_processing_pipeline import (
    concat_files,
    match_trajectories_to_poi,
    process_and_clean_gps,
    visualize_trajectories
)

def main():
    # Step 1: Merge raw GPS files
    input_files = [f'Day {i:02d}.csv' for i in range(2, 8)]
    merged_file = 'merged_all_days.csv'
    concat_files(file_list=input_files, output_file=merged_file)

    # Step 2: Match deviceId to POIs
    poi_file = 'Target-point.csv'
    matched_file = 'matched_trajectories.csv'
    match_trajectories_to_poi(
        trajectory_file=merged_file,
        poi_file=poi_file,
        output_file=matched_file,
        radius_meters=5000
    )

    # Step 3: Clean GPS data
    cleaned_file = 'cleaned_gps_data.csv'
    process_and_clean_gps(
        input_file=matched_file,
        output_file=cleaned_file
    )

    # Step 4: Visualize results
    visualize_trajectories(
        trajectory_file=cleaned_file,
        poi_file=poi_file,
        output_html='trajectories_map.html',
        max_vehicles=300
    )

if __name__ == '__main__':
    main()