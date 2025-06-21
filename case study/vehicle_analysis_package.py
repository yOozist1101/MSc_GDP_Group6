"""
Vehicle Analysis Package
Main module for vehicle trajectory analysis with dwell time detection and entry/exit analysis
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster


class VehicleAnalyzer:
    """Main class for vehicle trajectory analysis"""
    
    def __init__(self, data_dir='Data', output_dir='Output'):
        """
        Initialize the Vehicle Analyzer
        
        Args:
            data_dir (str): Directory containing input data files
            output_dir (str): Directory for output files
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Analysis parameters
        self.CENTER = (22.54238, 88.304)  # Center point for analysis
        self.ONE_MILE_RADIUS = 1600  # 1 mile ≈ 1600 meters
        self.ONE_MILE_KM = 1.6  # 1 mile ≈ 1.6 km
        self.RECT_BOUNDS = [[22.52, 88.28], [22.56, 88.32]]  # Analysis rectangle
        
        # Clustering parameters
        self.eps = 0.0003  # DBSCAN epsilon parameter
        self.min_samples = 5  # DBSCAN minimum samples
        
    def preprocess_data(self, input_file='cleaned_Kolkta_2_spots.csv', 
                       output_file='subset_vehicle_tracks.csv'):
        """
        Preprocess data by filtering vehicles in target area
        
        Args:
            input_file (str): Input CSV file name
            output_file (str): Output CSV file name
        """
        print("=== Step 1: Preprocessing Data ===")
        
        # Read data
        input_path = os.path.join(self.data_dir, input_file)
        df = pd.read_csv(input_path)
        df["gpsTime"] = pd.to_datetime(df["gpsTime"], errors="coerce")
        df = df.dropna(subset=["gpsTime", "latitude", "longitude"])
        
        # Filter vehicles in target area
        target_area = df[
            (df["latitude"] >= 22.534) & (df["latitude"] <= 22.546) &
            (df["longitude"] >= 88.294) & (df["longitude"] <= 88.308)
        ]
        target_ids = target_area["deviceId"].unique()
        
        # Extract full trajectories for these vehicles
        subset = df[df["deviceId"].isin(target_ids)].copy()
        subset = subset.sort_values(["deviceId", "gpsTime"])
        
        # Save results
        output_path = os.path.join(self.output_dir, output_file)
        subset.to_csv(output_path, index=False)
        print(f" Saved {len(subset):,} rows for {len(target_ids)} vehicles to {output_path}")
        
        return subset
    
    def analyze_dwell_time(self, input_file='subset_vehicle_tracks.csv',
                          filtered_output='filtered_dwell_clusters.csv',
                          pivot_output='dwell_time_by_hour.csv',
                          heatmap_output='dwell_time_heatmap_per_day.png',
                          map_output='dwell_map_per_day.html'):
        """
        Analyze dwell time using DBSCAN clustering
        
        Args:
            input_file (str): Input CSV file name
            filtered_output (str): Filtered clusters output file
            pivot_output (str): Pivot table output file
            heatmap_output (str): Heatmap output file
            map_output (str): Interactive map output file
        """
        print("=== Step 2: Analyzing Dwell Time ===")
        
        # Read data
        input_path = os.path.join(self.output_dir, input_file)
        df = pd.read_csv(input_path, parse_dates=["gpsTime"])
        df = df.sort_values(by=["deviceId", "gpsTime"]).dropna()
        df = df[["deviceId", "gpsTime", "latitude", "longitude", "deviceSpeed"]]
        
        # Detect stationary points
        df["is_stationary"] = df["deviceSpeed"] < 1.0
        df["date"] = df["gpsTime"].dt.date
        
        results = []
        
        # DBSCAN clustering for each device-date combination
        for (device, date), group in df.groupby(["deviceId", "date"]):
            stationary = group[group["is_stationary"]].copy()
            if len(stationary) < 5:
                continue
            
            coords = stationary[["latitude", "longitude"]].to_numpy()
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(coords)
            stationary["cluster"] = db.labels_
            stationary = stationary[stationary["cluster"] != -1].copy()
            if not stationary.empty:
                results.append(stationary)
        
        if not results:
            raise ValueError(" No stationary clusters found. Please check your data.")
        
        df_clustered = pd.concat(results)
        
        # Calculate cluster statistics
        cluster_times = df_clustered.groupby(["deviceId", "date", "cluster"]).agg(
            start_time=("gpsTime", "min"),
            end_time=("gpsTime", "max"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean")
        ).reset_index()
        
        cluster_times["duration_minutes"] = (
            (cluster_times["end_time"] - cluster_times["start_time"]).dt.total_seconds() / 60
        )
        
        # Filter for target area
        target_area = cluster_times[
            (cluster_times["latitude"] >= 22.534) & (cluster_times["latitude"] <= 22.546) &
            (cluster_times["longitude"] >= 88.294) & (cluster_times["longitude"] <= 88.308)
        ].copy()
        
        # Create pivot table
        target_area["hour"] = target_area["start_time"].dt.hour
        pivot_table = target_area.pivot_table(
            index="deviceId",
            columns="hour",
            values="duration_minutes",
            aggfunc="sum",
            fill_value=0
        )
        
        # Save results
        filtered_path = os.path.join(self.output_dir, filtered_output)
        pivot_path = os.path.join(self.output_dir, pivot_output)
        target_area.to_csv(filtered_path, index=False)
        pivot_table.to_csv(pivot_path)
        
        # Generate heatmap
        self._create_heatmap(pivot_table, heatmap_output)
        
        # Generate interactive map
        self._create_interactive_map(target_area, map_output)
        
        print(f" Dwell time analysis completed. Results saved to {self.output_dir}")
        
        return target_area, pivot_table
    
    def analyze_entry_exit(self, input_file='subset_vehicle_tracks.csv',
                          output_file='entry_exit_transition_durations.csv'):
        """
        Analyze entry/exit transitions
        
        Args:
            input_file (str): Input CSV file name
            output_file (str): Output CSV file name
        """
        print("=== Step 3: Analyzing Entry/Exit Transitions ===")
        
        # Read data
        input_path = os.path.join(self.output_dir, input_file)
        df = pd.read_csv(input_path, parse_dates=["gpsTime"])
        df = df.sort_values(by=["deviceId", "gpsTime"]).dropna()
        df["date"] = df["gpsTime"].dt.date
        
        # Define area checking functions
        def is_inside_area(lat, lon):
            return 22.52 <= lat <= 22.56 and 88.28 <= lon <= 88.32
        
        def is_within_1mile(lat, lon):
            return geodesic((lat, lon), self.CENTER).km <= self.ONE_MILE_KM
        
        def label_state(row):
            if is_inside_area(row["latitude"], row["longitude"]):
                return "in"
            elif is_within_1mile(row["latitude"], row["longitude"]):
                return "near"
            else:
                return "far"
        
        # Tag each point
        df["inside_area"] = df.apply(lambda row: is_inside_area(row["latitude"], row["longitude"]), axis=1)
        df["within_1mile"] = df.apply(lambda row: is_within_1mile(row["latitude"], row["longitude"]), axis=1)
        df["state"] = df.apply(label_state, axis=1)
        
        # Detect entry/exit transitions
        records = []
        
        for (device, date), group in df.groupby(["deviceId", "date"]):
            group = group.reset_index(drop=True)
            prev_state = group.loc[0, "state"]
            
            for i in range(1, len(group)):
                curr_state = group.loc[i, "state"]
                ts = group.loc[i, "gpsTime"]
                
                # Entry event: far -> in
                if prev_state == "far" and curr_state == "in":
                    entry_start = group.loc[i - 1, "gpsTime"]
                    entry_end = ts
                    duration = (entry_end - entry_start).total_seconds()
                    records.append({
                        "deviceId": device,
                        "date": date,
                        "type": "entry",
                        "from_time": entry_start,
                        "to_time": entry_end,
                        "duration_seconds": duration
                    })
                
                # Exit event: in -> far
                if prev_state == "in" and curr_state == "far":
                    exit_start = group.loc[i - 1, "gpsTime"]
                    exit_end = ts
                    duration = (exit_end - exit_start).total_seconds()
                    records.append({
                        "deviceId": device,
                        "date": date,
                        "type": "exit",
                        "from_time": exit_start,
                        "to_time": exit_end,
                        "duration_seconds": duration
                    })
                
                prev_state = curr_state
        
        # Save results
        result_df = pd.DataFrame(records)
        output_path = os.path.join(self.output_dir, output_file)
        result_df.to_csv(output_path, index=False)
        
        # Generate visualizations
        self._create_entry_exit_plots(result_df)
        
        print(f" Entry/exit analysis completed. Results saved to {output_path}")
        
        return result_df
    
    def _create_heatmap(self, pivot_table, filename):
        """Create dwell time heatmap"""
        plt.figure(figsize=(16, 8))
        sns.heatmap(pivot_table, cmap="YlOrRd", linewidths=0.5, linecolor="gray", 
                   annot=True, fmt=".1f")
        plt.title("Dwell Time Heatmap by Hour and Vehicle (Per-Day Clustering)", fontsize=16)
        plt.xlabel("Hour of Day")
        plt.ylabel("Vehicle ID")
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        print(f" Heatmap saved to {output_path}")
    
    def _create_interactive_map(self, target_area, filename):
        """Create interactive map with dwell clusters"""
        m = folium.Map(location=[22.54, 88.30], zoom_start=15)
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add analysis boundaries
        folium.Rectangle(
            bounds=self.RECT_BOUNDS,
            color='blue',
            fill=True,
            fill_opacity=0.05,
            popup="Analysis Rectangle"
        ).add_to(m)
        
        folium.Circle(
            location=self.CENTER,
            radius=self.ONE_MILE_RADIUS,
            color='red',
            fill=True,
            fill_opacity=0.05,
            popup="1 Mile Radius"
        ).add_to(m)
        
        folium.Marker(
            location=self.CENTER,
            tooltip="Center Point",
            icon=folium.Icon(color="gray", icon="info-sign")
        ).add_to(m)
        
        # Add dwell clusters
        def get_color(duration):
            if duration < 5:
                return "green"
            elif duration < 30:
                return "blue"
            elif duration < 90:
                return "yellow"
            elif duration < 300:
                return "orange"
            else:
                return "red"
        
        for _, row in target_area.iterrows():
            popup = f"{row['deviceId']}<br>{row['start_time']}<br>{row['duration_minutes']:.1f} min"
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                color=get_color(row["duration_minutes"]),
                fill=True,
                fill_opacity=0.8,
                popup=popup
            ).add_to(marker_cluster)
        
        # Save map
        output_path = os.path.join(self.output_dir, filename)
        m.save(output_path)
        print(f" Interactive map saved to {output_path}")
    
    def _create_entry_exit_plots(self, result_df):
        """Create entry/exit analysis plots"""
        # Duration distribution histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=result_df,
            x="duration_seconds",
            hue="type",
            bins=40,
            kde=True,
            palette={"entry": "#1f77b4", "exit": "#d62728"}
        )
        plt.title("Entry/Exit Transition Durations (in seconds)")
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, "entry_exit_duration_hist.png")
        plt.savefig(output_path)
        plt.close()
        
        # Filtered duration (0-2500 seconds)
        filtered_df = result_df[result_df["duration_seconds"] <= 2500]
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=filtered_df,
            x="duration_seconds",
            hue="type",
            bins=30,
            kde=True,
            palette={"entry": "#1f77b4", "exit": "#d62728"}
        )
        plt.title("Entry/Exit Transition Durations (0–2500 seconds)")
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, "entry_exit_duration_0_2500s.png")
        plt.savefig(output_path)
        plt.close()
        
        # Hourly analysis
        df_hourly = result_df.copy()
        df_hourly["hour"] = pd.to_datetime(df_hourly["from_time"]).dt.hour
        df_hourly["weekday"] = pd.to_datetime(df_hourly["from_time"]).dt.weekday
        df_hourly["day_type"] = df_hourly["weekday"].apply(lambda x: "Weekday" if x < 5 else "Weekend")
        
        # Boxplot by hour
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=df_hourly,
            x="hour",
            y="duration_seconds",
            hue="type",
            palette={"entry": "#1f77b4", "exit": "#d62728"}
        )
        plt.title("Entry/Exit Duration Distribution by Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("Duration (seconds)")
        plt.grid(True)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, "entry_exit_duration_by_hour_boxplot.png")
        plt.savefig(output_path)
        plt.close()
        
        # Barplot by hour
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df_hourly,
            x="hour",
            y="duration_seconds",
            hue="type",
            estimator="mean",
            ci="sd",
            palette={"entry": "#1f77b4", "exit": "#d62728"}
        )
        plt.title("Average Entry/Exit Duration by Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("Average Duration (seconds)")
        plt.grid(True)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, "entry_exit_duration_by_hour_barplot.png")
        plt.savefig(output_path)
        plt.close()
        
        # Weekday vs Weekend analysis
        g = sns.FacetGrid(
            df_hourly, row="day_type", hue="type", height=5, aspect=2, sharey=False,
            palette={"entry": "#1f77b4", "exit": "#d62728"}
        )
        g.map(sns.barplot, "hour", "duration_seconds", estimator="mean", ci="sd", order=range(24))
        g.add_legend()
        g.set_axis_labels("Hour of Day", "Average Duration (seconds)")
        g.fig.suptitle("Average Entry/Exit Duration by Hour (Weekday vs Weekend)", fontsize=16, y=1.05)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, "entry_exit_by_hour_weektype_barplot.png")
        g.savefig(output_path)
        plt.close()
        
        print(f" Entry/exit plots saved to {self.output_dir}")
    
    def run_full_analysis(self, input_file='cleaned_Kolkta_2_spots.csv'):
        """
        Run complete analysis pipeline
        
        Args:
            input_file (str): Input CSV file name
        """
        print("=== Starting Complete Vehicle Analysis ===")
        
        # Step 1: Preprocess data
        subset_df = self.preprocess_data(input_file=input_file)
        
        # Step 2: Analyze dwell time
        dwell_results, pivot_table = self.analyze_dwell_time()
        
        # Step 3: Analyze entry/exit transitions
        entry_exit_results = self.analyze_entry_exit()
        
        print("=== Analysis Complete ===")
        print(f"All results saved to: {self.output_dir}")
        
        return {
            'subset_data': subset_df,
            'dwell_results': dwell_results,
            'pivot_table': pivot_table,
            'entry_exit_results': entry_exit_results
        }


def main():
    """Main function to run the vehicle analysis"""
    # Initialize analyzer
    analyzer = VehicleAnalyzer(data_dir='Data', output_dir='Output')
    
    # Run complete analysis
    results = analyzer.run_full_analysis(input_file='cleaned_Kolkta_2_spots.csv')
    
    print("\n=== Analysis Summary ===")
    print(f"Processed {len(results['subset_data'])} GPS records")
    print(f"Found {len(results['dwell_results'])} dwell clusters")
    print(f"Detected {len(results['entry_exit_results'])} entry/exit transitions")


if __name__ == "__main__":
    main()
