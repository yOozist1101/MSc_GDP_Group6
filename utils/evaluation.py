import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import os
from collections import defaultdict
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from folium.plugins import MarkerCluster, HeatMap
from datetime import time
from geopy.distance import geodesic
from matplotlib.ticker import MaxNLocator

class TrafficAnalysisSystem:
    """Comprehensive traffic data analysis and visualization system"""
    
    def __init__(self, city_bounds=None, input_dir="Data", output_dir="Output"):
        self.city_bounds = city_bounds or {
            'min_lat': -90, 'max_lat': 90,
            'min_lon': -180, 'max_lon': 180
        }
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.vehicle_class_labels = {
            1: 'Light passenger van/minibus',
            2: 'Medium truck',
            3: 'Heavy truck',
            4: 'Bus',
            5: 'Ambulance', 
            6: 'Pick-up',
            7: 'Tractor',
            8: 'Tricycle',
            9: 'Scooter/motorbike'
        }
        plt.style.use('seaborn-v0_8')
        
        # Create output directory structure
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self, file_name, chunksize=50000):
        """Load and preprocess traffic data"""
        try:
            file_path = os.path.join(self.input_dir, file_name)
            chunks = pd.read_csv(file_path, parse_dates=['gpsTime'], chunksize=chunksize)
            return pd.concat(chunks)
        except Exception as e:
            print(f"Error loading data from {file_name}: {e}")
            return None

    def analyze_and_save_results(self, df, output_subdir):
        """Enhanced version of the original evaluation analysis"""
        # Create output subdirectory
        full_output_dir = os.path.join(self.output_dir, output_subdir)
        os.makedirs(full_output_dir, exist_ok=True)
        
        # 1. Substate Distribution Statistics
        substate_counts = df['substate'].value_counts().reset_index()
        substate_counts.columns = ['substate', 'count']
        print(f"\nSubstate distribution ({output_subdir}):")
        print(substate_counts)
        substate_counts.to_csv(os.path.join(full_output_dir, 'substate_counts.csv'), index=False)

        # 2. Substate Percentage Distribution
        substate_percent = df['substate'].value_counts(normalize=True) * 100
        print(f"\nSubstate percentage distribution ({output_subdir}):")
        print(substate_percent.round(2).astype(str) + '%')
        substate_percent.to_csv(os.path.join(full_output_dir, 'substate_percent.csv'), index=True)

        # Visualization 1: Substate distribution pie chart
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12    
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['legend.title_fontsize'] = 12

        # Get sorted data for pie chart
        substate_percent_sorted = substate_percent.sort_values(ascending=False)
        labels = substate_percent_sorted.index.tolist()
        sizes = substate_percent_sorted.values.tolist()
        
        # Create color palette
        base = sns.color_palette("Blues_r", len(labels))
        palette = [(r*0.6 + 0.4, g*0.6 + 0.4, b*0.6 + 0.4) for (r, g, b) in base]

        fig, ax = plt.subplots(figsize=(8, 6))

        # Create pie chart (without labels on the chart, show in legend)
        wedges, _ = ax.pie(
            sizes,
            colors=palette,
            startangle=140,
            wedgeprops={'edgecolor':'white', 'linewidth':1}
        )

        # Keep pie chart as perfect circle
        ax.axis('equal')

        # Add separate legend
        ax.legend(
            wedges, labels, title="Legend",
            loc="center left", bbox_to_anchor=(1, 0, 0.3, 1),
            frameon=False, prop={'family':'Times New Roman'}
        )
        ax.set(aspect="equal")
        plt.xlim(-1.2, 2.2)
        plt.ylim(-1.2, 1.2)
        plt.tight_layout()
        plt.savefig(os.path.join(full_output_dir, 'substate_pie_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Visualization 2: Temporal distribution of substates
        if 'gpsTime' in df.columns:
            df['hour'] = pd.to_datetime(df['gpsTime']).dt.hour
            plt.figure(figsize=(14, 7))
            sns.countplot(data=df, x='hour', hue='substate', 
                         hue_order=df['substate'].value_counts().index[:5]) # limit to show top 5 substates
            plt.title('Hourly Substate Distribution')
            plt.savefig(os.path.join(full_output_dir, 'hourly_substate.png'), dpi=300)
            plt.close()

        # Duration metrics for all substates  
        if 'duration_secs' in df.columns and 'substate' in df.columns:
            # Use all states without filtering
            filtered_df = df.copy()

            if not filtered_df.empty:
                # Ensure data is sorted by device and time (important!)
                if 'deviceId' in filtered_df.columns:
                    filtered_df = filtered_df.sort_values(['deviceId', 'gpsTime']).reset_index(drop=True)
                else:
                    filtered_df = filtered_df.sort_values('gpsTime').reset_index(drop=True)
        
                # Mark new substate events by grouping by device
                if 'deviceId' in filtered_df.columns:
                    filtered_df['new_event'] = (
                        filtered_df.groupby('deviceId')['substate']
                        .apply(lambda x: (x != x.shift()).astype(int))
                        .reset_index(drop=True)
                    )
                else:
                    # If no device ID, mark globally
                    filtered_df['new_event'] = (filtered_df['substate'] != filtered_df['substate'].shift()).astype(int)

                # Generate event numbers using cumsum
                filtered_df['event_id'] = filtered_df['new_event'].cumsum()

                # Keep only one row per event (first row) to avoid duplicate duration calculations
                unique_events = filtered_df.drop_duplicates(subset='event_id', keep='first')
        
                # Verify deduplication effect
                print(f"Original rows: {len(filtered_df)}")
                print(f"Unique events: {len(unique_events)}")
                print(f"Duplicate events removed: {len(filtered_df) - len(unique_events)}")
            
                # Filter out extremely small values
                MIN_DURATION_THRESHOLD = 5  # Example: filter out records less than 5 seconds
            
                # Statistics before filtering
                print(f"\nBefore filtering minimum duration:")
                print(f"Total events: {len(unique_events)}")
                print(f"Duration range: {unique_events['duration_secs'].min():.2f} - {unique_events['duration_secs'].max():.2f} seconds")
            
                # Apply minimum duration filter
                unique_events_filtered = unique_events[unique_events['duration_secs'] >= MIN_DURATION_THRESHOLD]
            
                # Statistics after filtering
                removed_count = len(unique_events) - len(unique_events_filtered)
                print(f"\nAfter filtering (minimum {MIN_DURATION_THRESHOLD} seconds):")
                print(f"Events remaining: {len(unique_events_filtered)}")
                print(f"Events removed: {removed_count} ({removed_count/len(unique_events)*100:.1f}%)")
                if len(unique_events_filtered) > 0:
                    print(f"New duration range: {unique_events_filtered['duration_secs'].min():.2f} - {unique_events_filtered['duration_secs'].max():.2f} seconds")

                # Check if data remains after filtering
                if len(unique_events_filtered) == 0:
                    print(f"\nWarning: No events remain after filtering. Consider lowering the minimum duration threshold.")
                else:
                    # Group statistics for each substate's duration (using filtered data)
                    duration_stats = unique_events_filtered.groupby('substate')['duration_secs'].agg(
                        ['mean', 'median', 'std', 'count', 'sum']
                    ).sort_values('count', ascending=False)

                    print(f"\nAll States Duration Statistics (seconds) - After Filtering ({output_subdir}):")
                    print(duration_stats.round(2))

                    # Save as CSV
                    duration_stats.to_csv(os.path.join(full_output_dir, 'all_states_duration_stats_filtered.csv'))

        # State transition matrix
        transitions = defaultdict(int)
        for _, group in df.sort_values('gpsTime').groupby('deviceId'): 
            states = group['substate'].values
            for i in range(len(states)-1):
                transitions[(states[i], states[i+1])] += 1

        # Create dataframe
        trans_matrix = pd.DataFrame(
            index=df['substate'].unique(),
            columns=df['substate'].unique()
        ).fillna(0)

        for (s1, s2), count in transitions.items():
            trans_matrix.at[s1, s2] = count

        # Save transition matrix
        trans_matrix.to_csv(os.path.join(full_output_dir, 'state_transition_matrix.csv'))
        
        # Visualization 3: State Transition Matrix
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(
            trans_matrix, 
            annot=True, 
            fmt='.0f',  # integer format
            cmap='Blues',
            annot_kws={'size': 8}, 
            cbar_kws={'shrink': 0.8}  
            )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        plt.title('State Transition Matrix (Counts)')
        plt.tight_layout()  
        plt.savefig(os.path.join(full_output_dir, 'state_transition.png'), dpi=300, bbox_inches='tight')  
        plt.close()

    def analyze_vehicle_classes(self, df, output_file='vehicle_classes.csv'):
        """Analyze vehicle class distribution"""
        df['vehicleClass'] = pd.to_numeric(df['vehicleClass'], errors='coerce')
        valid_df = df[df['vehicleClass'].between(1, 9)]
        
        counts = valid_df['vehicleClass'].value_counts().sort_index()
        percentages = (counts / counts.sum() * 100).round(2)
        
        result = pd.DataFrame({
            'Count': counts,
            'Percentage': percentages,
            'Description': counts.index.map(self.vehicle_class_labels)
        })
        
        result.to_csv(os.path.join(self.output_dir, output_file))
        print(f"Vehicle class analysis saved to {output_file}")
        return result

    def create_traffic_map(self, df, substate, output_file, 
                         banned_hours=None, sample_size=1000):
        """Generate interactive traffic map for specific substate"""
        # Filter and clean data
        data = df[df['substate'] == substate].copy()
        data = data.dropna(subset=['latitude', 'longitude', 'duration_secs'])
        data = data[
            (data['latitude'].between(self.city_bounds['min_lat'], self.city_bounds['max_lat'])) & 
            (data['longitude'].between(self.city_bounds['min_lon'], self.city_bounds['max_lon']))
        ]
        data['duration_mins'] = data['duration_secs'] / 60
        
        # Create base map
        m = folium.Map(
            location=[data['latitude'].median(), data['longitude'].median()],
            zoom_start=12,
            tiles='CartoDB positron',
            control_scale=True
        )
        
        # Add heatmap
        HeatMap(
            data[['latitude', 'longitude', 'duration_mins']].values,
            radius=15,
            blur=10,
            min_opacity=0.5,
            gradient={0.3: 'blue', 0.5: 'lime', 0.7: 'yellow', 1: 'red'}
        ).add_to(m)
        
        # Add markers
        marker_cluster = MarkerCluster(options={'maxClusterRadius': 30})
        for _, row in data.sample(min(sample_size, len(data))).iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4 + np.log1p(row['duration_mins']),
                color='#E74C3C',
                fill=True,
                fill_color='#F1C40F',
                popup=f"Duration: {row['duration_mins']:.1f} mins"
            ).add_to(marker_cluster)
        
        # Add controls
        folium.LayerControl().add_to(m)
        folium.plugins.Fullscreen().add_to(m)
        
        output_path = os.path.join(self.output_dir, output_file)
        m.save(output_path)
        print(f"Traffic map saved to {output_path}")
        return output_path

    def plot_weekday_comparison(self, df, top_n=4, output_file='weekday_comparison.png'):
        """Compare weekday vs weekend patterns"""
        df['day_type'] = df['gpsTime'].dt.dayofweek.apply(
            lambda x: 'Weekend' if x >= 5 else 'Weekday'
        )
        
        top_substates = df['substate'].value_counts().nlargest(top_n).index
        stats = df[df['substate'].isin(top_substates)].groupby(
            ['substate', 'day_type']
        ).size().unstack()
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(stats))
        width = 0.35
        
        ax.bar(x - width/2, stats['Weekday'], width, label='Weekday')
        ax.bar(x + width/2, stats['Weekend'], width, label='Weekend')
        
        ax.set_title('Substate Distribution: Weekday vs Weekend')
        ax.set_xticks(x)
        ax.set_xticklabels(stats.index)
        ax.legend()
        
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Weekday comparison saved to {output_path}")
        return output_path

    def plot_normalized_comparison(self, df, top_n=4, output_file='normalized_comparison.png'):
        """Compare normalized weekday vs weekend patterns"""
        df['day_type'] = df['gpsTime'].dt.dayofweek.apply(
            lambda x: 'Weekend' if x >= 5 else 'Weekday'
        )
        
        top_substates = df['substate'].value_counts().nlargest(top_n).index
        df = df[df['substate'].isin(top_substates)]
        
        # Calculate proportions
        norm_stats = (df.groupby(['substate', 'day_type']).size() / 
                    df.groupby('day_type').size()).unstack()
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(norm_stats))
        width = 0.35
        
        ax.bar(x - width/2, norm_stats['Weekday'], width, label='Weekday', color='#3498db')
        ax.bar(x + width/2, norm_stats['Weekend'], width, label='Weekend', color='#e74c3c')
        
        ax.set_title('Normalized Substate Distribution (Weekday vs Weekend)', pad=20)
        ax.set_ylabel('Proportion')
        ax.set_xticks(x)
        ax.set_xticklabels(norm_stats.index)
        ax.legend()
        
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path, bbox_inches='tight', dpi=120)
        plt.close()
        print(f"Normalized comparison saved to {output_path}")
        return output_path

    def calculate_violation_ratio(self, df, banned_hours, required_substate='normal_driving', 
                                min_points=2, violation_threshold=0.5, output_file='violation_ratio.png'):
        """Calculate percentage of vehicles with points during banned hours"""
        print(f"Analyzing {required_substate} violations...")
        
        try:
            # Filter data
            data = df[df['substate'] == required_substate].copy()
            print(f"Found {len(data)} {required_substate} records")
            
            # Filter for city area
            in_city = data[
                (data['latitude'].between(self.city_bounds['min_lat'], self.city_bounds['max_lat'])) & 
                (data['longitude'].between(self.city_bounds['min_lon'], self.city_bounds['max_lon']))
            ]
            print(f"Records within city bounds: {len(in_city)}")
            
            if len(in_city) == 0:
                print("No valid data found")
                return None
            
            # Mark banned time points
            in_city['time'] = in_city['gpsTime'].dt.time
            in_city['is_banned'] = in_city['time'].between(*banned_hours)
            
            # Calculate violation ratio per vehicle
            vehicle_stats = in_city.groupby('deviceId')['is_banned'].agg(
                total_points='count',
                banned_points='sum'
            ).query(f'total_points >= {min_points}').reset_index()
            
            vehicle_stats['banned_ratio'] = vehicle_stats['banned_points'] / vehicle_stats['total_points']
            vehicle_stats['is_violator'] = vehicle_stats['banned_ratio'] > violation_threshold
            
            # Calculate results
            violators = vehicle_stats[vehicle_stats['is_violator']]
            valid_vehicles = len(vehicle_stats)
            violation_ratio = len(violators) / valid_vehicles if valid_vehicles > 0 else 0
            
            print(f"\nViolation Statistics ({required_substate}):")
            print(f"Valid vehicles (≥{min_points} points): {valid_vehicles}")
            print(f"Violating vehicles (>{(violation_threshold*100):.0f}% points): {len(violators)}")
            print(f"Violation ratio: {violation_ratio:.2%}")
            
            # Create pie chart visualization
            plt.figure(figsize=(8, 6))
            
            # Pie chart data
            sizes = [valid_vehicles-len(violators), len(violators)]
            labels = [f'Compliant', f'Violating']
            colors = ['#4CAF50', '#F44336']
            explode = (0, 0.1)  # Highlight violating vehicles
            
            plt.pie(sizes, labels=labels, colors=colors, explode=explode,
                    autopct='%1.1f%%', shadow=True, startangle=90,
                    wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
            
            plt.title(f'Traffic Violation Ratio\n({required_substate} state)', pad=20)
            
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, dpi=120, bbox_inches='tight', transparent=True)
            plt.close()
            print(f"Visualization saved to: {output_path}")
            
            return {
                'total_vehicles': valid_vehicles,
                'violators': len(violators),
                'violation_ratio': violation_ratio
            }
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return None

    def calculate_mumbai_violations(self, df, banned_windows, required_substate='normal_driving', 
                                  min_points=3, violation_threshold=0.5, output_file='peak_hour_violations.png'):
        """Analyze traffic violations with multiple banned windows"""
        print(f"Analyzing {required_substate} violations with multiple banned windows...")
        
        try:
            # Filter data
            data = df[df['substate'] == required_substate].copy()
            print(f"Found {len(data)} {required_substate} records")
            
            # Filter for city area
            in_city = data[
                (data['latitude'].between(self.city_bounds['min_lat'], self.city_bounds['max_lat'])) & 
                (data['longitude'].between(self.city_bounds['min_lon'], self.city_bounds['max_lon']))
            ]
            print(f"Records within city bounds: {len(in_city)}")
            
            if len(in_city) == 0:
                print("No valid data found within city boundaries")
                return None
            
            # Mark banned time points (check all windows)
            in_city['time'] = in_city['gpsTime'].dt.time
            in_city['is_banned'] = False
            for window in banned_windows:
                in_city['is_banned'] |= in_city['time'].between(*window)
            
            # Calculate violation ratio per vehicle
            vehicle_stats = in_city.groupby('deviceId')['is_banned'].agg(
                total_points='count',
                banned_points='sum'
            ).query(f'total_points >= {min_points}').reset_index()
            
            vehicle_stats['banned_ratio'] = vehicle_stats['banned_points'] / vehicle_stats['total_points']
            vehicle_stats['is_violator'] = vehicle_stats['banned_ratio'] > violation_threshold
            
            # Calculate results
            violators = vehicle_stats[vehicle_stats['is_violator']]
            valid_vehicles = len(vehicle_stats)
            violation_ratio = len(violators) / valid_vehicles if valid_vehicles > 0 else 0
            
            print(f"\nViolation Statistics ({required_substate}):")
            print(f"Valid vehicles (≥{min_points} points): {valid_vehicles:,}")
            print(f"Violating vehicles (>{(violation_threshold*100):.0f}% points in banned windows): {len(violators):,}")
            print(f"Violation ratio: {violation_ratio:.2%}")
            
            # Create professional pie chart
            plt.figure(figsize=(10, 7))
            
            # Data preparation
            compliant = valid_vehicles - len(violators)
            sizes = [compliant, len(violators)]
            labels = [
                f'Compliant\n{compliant:,} vehicles\n({(1-violation_ratio):.1%})',
                f'Violating\n{len(violators):,} vehicles\n({violation_ratio:.1%})'
            ]
            colors = ['#27ae60', '#e74c3c']  # Green/Red palette
            explode = (0, 0.08)  # Slight emphasis on violators
            
            # Create pie chart
            wedges, texts, autotexts = plt.pie(
                sizes, labels=labels, colors=colors, explode=explode,
                autopct=lambda p: '',  # We'll add custom percentage labels
                shadow=True, startangle=90,
                wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
                textprops={'fontsize': 11, 'fontweight': 'bold'}
            )
            
            # Add detailed title
            banned_hours_str = " | ".join([f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}" 
                                         for start, end in banned_windows])
            plt.title(
                f'Traffic Violation Analysis\n'
                f'({banned_hours_str} | {required_substate} only)',
                pad=25, fontsize=14, fontweight='bold'
            )
            
            # Save high-quality output
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Visualization saved to: {output_path}")
            
            return {
                'total_vehicles': valid_vehicles,
                'violators': len(violators),
                'violation_ratio': violation_ratio
            }
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return None

    def classify_trajectory_directions(self, df, targets_df):
        """Classify trajectories as inbound or outbound relative to target points"""
        print("Classifying trajectory directions...")
        
        try:
            # Prepare output results
            results = []
            
            # Process each target point
            for _, target in targets_df.iterrows():
                target_name = target['Target']
                target_coord = (target['latitude'], target['longitude'])
                
                # Filter trajectories for this target
                df_target = df[df['Target'] == target_name]
                
                # Process each vehicle's trajectory
                for device_id, group in df_target.groupby("deviceId"):
                    group_sorted = group.sort_values("gpsTime")
                    start = (group_sorted.iloc[0]['latitude'], group_sorted.iloc[0]['longitude'])
                    end = (group_sorted.iloc[-1]['latitude'], group_sorted.iloc[-1]['longitude'])
                    
                    # Calculate distances
                    dist_start = geodesic(start, target_coord).meters
                    dist_end = geodesic(end, target_coord).meters
                    
                    # Determine direction
                    direction = "Inbound" if dist_end < dist_start else "Outbound"
                    
                    results.append({
                        "deviceId": device_id,
                        "Target": target_name,
                        "Start_Lat": start[0],
                        "Start_Lon": start[1],
                        "End_Lat": end[0],
                        "End_Lon": end[1],
                        "Dist_Start(m)": round(dist_start, 2),
                        "Dist_End(m)": round(dist_end, 2),
                        "Direction": direction
                    })
            
            # Convert to DataFrame
            result_df = pd.DataFrame(results)
            print(f"Successfully classified {len(result_df)} trajectories")
            return result_df
            
        except Exception as e:
            print(f"Error classifying trajectories: {str(e)}")
            return None

    def merge_direction_info(self, trajectory_df, direction_df, output_file='trajectory_with_direction.csv'):
        """Merge direction classification with original trajectory data"""
        try:
            merged_df = trajectory_df.merge(
                direction_df[['deviceId', 'Direction']], 
                on='deviceId', 
                how='left'
            )
            
            output_path = os.path.join(self.output_dir, output_file)
            merged_df.to_csv(output_path, index=False)
            print(f"Saved merged data to {output_path}")
            
            return merged_df
        except Exception as e:
            print(f"Error merging direction info: {str(e)}")
            return None

    def analyze_direction_distribution(self, df, output_prefix='direction'):
        """Analyze and visualize vehicle direction distribution"""
        print("Analyzing vehicle direction distribution...")
        
        try:
            # Check required columns
            required_columns = ['Direction', 'latitude', 'longitude']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                return None
            
            # 1. Direction statistics
            direction_counts = df['Direction'].value_counts()
            total = direction_counts.sum()
            
            print("\nDirection statistics:")
            for direction, count in direction_counts.items():
                percentage = (count / total) * 100
                print(f"• {direction}: {count} ({percentage:.1f}%)")
            
            # 2. Create pie chart
            plt.figure(figsize=(10, 8))
            plt.pie(
                direction_counts,
                labels=direction_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=['#4CAF50', '#F44336', '#2196F3'],
                textprops={'fontsize': 14},
                wedgeprops={'edgecolor': 'white', 'linewidth': 2}
            )
            plt.title('Vehicle Direction Distribution\n\n', fontsize=16, pad=20)
            
            pie_chart_file = os.path.join(self.output_dir, f"{output_prefix}_pie.png")
            plt.savefig(pie_chart_file, dpi=120, bbox_inches='tight', transparent=True)
            print(f"Saved pie chart: {pie_chart_file}")
            plt.close()
            
            # 3. Create direction map
            print("Creating direction distribution map...")
            
            # Calculate center point
            center_lat = df['latitude'].median()
            center_lon = df['longitude'].median()
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=13,
                tiles='cartodbpositron',
                control_scale=True
            )
            
            # Add direction clusters
            def add_direction_cluster(map_obj, data, direction, color):
                if len(data) == 0:
                    return
                    
                cluster = MarkerCluster(
                    name=f"{direction} Vehicles",
                    options={
                        'maxClusterRadius': 40,
                        'spiderfyDistanceMultiplier': 1.5
                    }
                ).add_to(map_obj)
                
                # Add markers
                for _, row in data.sample(min(1000, len(data))).iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=6,
                        color=color,
                        weight=1,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=f"""
                            <div style="width:250px">
                                <b>{direction} Vehicle</b><br>
                                <hr style="margin:5px 0">
                                ID: {row.get('deviceId','N/A')}<br>
                                Time: {row.get('gpsTime','N/A')}<br>
                                Speed: {row.get('deviceSpeed',0):.1f} km/h
                            </div>
                        """
                    ).add_to(cluster)
                
                # Add legend
                folium.map.Marker(
                    [center_lat + 0.02, center_lon],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:14px; color:{color}">■ {direction} Vehicles</div>'
                    )
                ).add_to(map_obj)
            
            # Add direction data to map
            for direction, color in [('Inbound', '#4CAF50'), ('Outbound', '#F44336')]:
                if direction in direction_counts.index:
                    add_direction_cluster(m, df[df['Direction'] == direction], direction, color)
            
            # Add controls
            folium.LayerControl(position='topright').add_to(m)
            folium.plugins.Fullscreen(position='topleft').add_to(m)
            folium.plugins.MousePosition(position='bottomleft').add_to(m)
            
            # Save map
            map_file = os.path.join(self.output_dir, f"{output_prefix}_map.html")
            m.save(map_file)
            print(f"Saved direction map: {map_file}")
            
            return {
                'direction_counts': direction_counts,
                'pie_chart': pie_chart_file,
                'map_file': map_file
            }
            
        except Exception as e:
            print(f"Error analyzing direction distribution: {str(e)}")
            return None

    def analyze_direction_timing(self, df, output_prefix='direction_timing'):
        """Analyze inbound/outbound vehicle distribution by time"""
        print("Analyzing inbound/outbound time distribution...")
        
        try:
            # Check required columns
            required_columns = ['Direction', 'gpsTime']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                return None
            
            # 1. Extract time features
            df['hour'] = df['gpsTime'].dt.hour
            time_bins = [0,6,9,12,15,18,21,24]
            time_labels = ['Night (0-6)', 'Morning Peak (6-9)', 'Late Morning (9-12)', 
                         'Noon (12-15)', 'Afternoon (15-18)', 'Evening Peak (18-21)', 
                         'Night (21-24)']
            df['time_period'] = pd.cut(df['hour'], bins=time_bins, labels=time_labels, right=False)
            
            # 2. Time distribution statistics
            time_stats = df.groupby(['time_period', 'Direction']).size().unstack().fillna(0)
            print("\nTime period statistics:")
            print(time_stats)
            
            # 3. Create 24-hour distribution plot
            plt.figure(figsize=(14, 10))
            
            # Hourly stats
            hourly_stats = df.groupby(['hour', 'Direction']).size().unstack().fillna(0)
            
            # Line plot
            plt.subplot(2, 1, 1)
            for direction in ['Inbound', 'Outbound']:
                if direction in hourly_stats.columns:
                    plt.plot(hourly_stats.index, hourly_stats[direction], 
                            marker='o', markersize=8, label=direction, 
                            linewidth=2.5, alpha=0.8)
            
            plt.title('24-Hour Inbound/Outbound Traffic Flow', fontsize=16, pad=20)
            plt.xlabel('Hour of Day', fontsize=12)
            plt.ylabel('Number of Vehicles', fontsize=12)
            plt.xticks(range(0, 24))
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(fontsize=12, frameon=True)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.ylim(0, hourly_stats.max().max()*1.1)
            
            # 4. Stacked bar chart by time period
            plt.subplot(2, 1, 2)
            time_stats.plot(kind='bar', stacked=True, 
                          color=['#3498db', '#e74c3c'], 
                          edgecolor='white', linewidth=1, 
                          width=0.85, ax=plt.gca())
            
            plt.title('Inbound/Outbound Distribution by Time Period', fontsize=16, pad=20)
            plt.xlabel('Time Period', fontsize=12)
            plt.ylabel('Number of Vehicles', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.6, axis='y')
            plt.legend(title='Direction', fontsize=10, frameon=True)
            
            plt.tight_layout()
            time_dist_file = os.path.join(self.output_dir, f"{output_prefix}.png")
            plt.savefig(time_dist_file, dpi=120, bbox_inches='tight', facecolor='white')
            print(f"Saved time distribution plot: {time_dist_file}")
            plt.close()
            
            # 5. Heatmap (Hour × Direction)
            plt.figure(figsize=(12, 6))
            pivot_hour = df.pivot_table(index='hour', columns='Direction', 
                                      aggfunc='size', fill_value=0)
            
            sns.heatmap(pivot_hour, cmap='YlOrRd', annot=True, fmt='d',
                       linewidths=0.5, cbar_kws={'label': 'Vehicle Count'},
                       annot_kws={'size': 10})
            
            plt.title('Hourly Inbound/Outbound Heatmap', fontsize=16, pad=20)
            plt.xlabel('Direction', fontsize=12)
            plt.ylabel('Hour of Day', fontsize=12)
            
            heatmap_file = os.path.join(self.output_dir, f"{output_prefix}_heatmap.png")
            plt.savefig(heatmap_file, dpi=120, bbox_inches='tight', facecolor='white')
            print(f"Saved heatmap: {heatmap_file}")
            plt.close()
            
            # 6. Peak hour analysis
            peak_hours = {}
            if 'Inbound' in hourly_stats.columns:
                peak_inbound = hourly_stats['Inbound'].idxmax()
                peak_hours['Inbound'] = peak_inbound
                print(f"\nPeak Inbound Hour: {peak_inbound}:00-{peak_inbound+1}:00")
            
            if 'Outbound' in hourly_stats.columns:
                peak_outbound = hourly_stats['Outbound'].idxmax()
                peak_hours['Outbound'] = peak_outbound
                print(f"Peak Outbound Hour: {peak_outbound}:00-{peak_outbound+1}:00")
            
            return {
                'time_stats': time_stats,
                'hourly_stats': hourly_stats,
                'peak_hours': peak_hours,
                'time_dist_file': time_dist_file,
                'heatmap_file': heatmap_file
            }
            
        except Exception as e:
            print(f"Error analyzing direction timing: {str(e)}")
            return None


def main():
    """Main function to execute comprehensive traffic analysis"""
    print("Starting Comprehensive Traffic Analysis System")
    print("=" * 50)
    
    # Initialize analyzer with flexible city bounds
    analyzer = TrafficAnalysisSystem(
        city_bounds={
            'min_lat': -90, 'max_lat': 90,
            'min_lon': -180, 'max_lon': 180
        },
        input_dir="Data",
        output_dir="Output"
    )
    
    try:
        # 1. Load main dataset
        print("\n1. Loading sample dataset...")
        df = analyzer.load_data("sample_data_with_substate.csv")
        
        if df is None:
            print("Error: Could not load sample_data_with_substate.csv")
            print("Please ensure the file exists in the Data/ directory")
            return
        
        print(f"Loaded sample data: {len(df)} records")
        
        # 2. Basic substate analysis
        print("\n2. Performing basic substate analysis...")
        analyzer.analyze_and_save_results(df, 'sample_results')
        
        # 3. Vehicle class analysis (if column exists)
        print("\n3. Analyzing vehicle classes...")
        if 'vehicleClass' in df.columns:
            analyzer.analyze_vehicle_classes(df, 'vehicle_classes.csv')
        else:
            print("Vehicle class column not found, skipping vehicle class analysis")
        
        # 4. Create traffic maps for different substates
        print("\n4. Creating traffic maps...")
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Get top substates for mapping
            top_substates = df['substate'].value_counts().head(3).index.tolist()
            
            for substate in top_substates:
                if len(df[df['substate'] == substate]) > 100:  # Only map if enough data
                    analyzer.create_traffic_map(df, substate, f'{substate}_map.html')
        else:
            print("Location columns not found, skipping map creation")
        
        # 5. Weekday vs Weekend analysis
        print("\n5. Analyzing weekday vs weekend patterns...")
        if 'gpsTime' in df.columns:
            analyzer.plot_weekday_comparison(df, output_file='weekday_comparison.png')
            analyzer.plot_normalized_comparison(df, output_file='normalized_comparison.png')
        else:
            print("Time column not found, skipping temporal analysis")
        
        # 6. Violation analysis
        print("\n6. Analyzing traffic violations...")
        if 'gpsTime' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
            # Single banned hour window
            analyzer.calculate_violation_ratio(
                df, 
                banned_hours=(time(6,0), time(22,0)),
                output_file='single_window_violations.png'
            )
            
            # Multiple banned hour windows (peak hours)
            analyzer.calculate_mumbai_violations(
                df, 
                banned_windows=[(time(8,0), time(11,0)), (time(17,0), time(21,0))],
                output_file='peak_hour_violations.png'
            )
        else:
            print("Required columns for violation analysis not found, skipping")
        
        # 7. Direction analysis (if target points are available)
        print("\n7. Analyzing trajectory directions...")
        try:
            targets_df = analyzer.load_data("Target-point.csv")
            if targets_df is not None and 'Target' in df.columns:
                print("Target points and trajectory data found, performing direction analysis...")
                
                # Classify directions
                direction_df = analyzer.classify_trajectory_directions(df, targets_df)
                
                if direction_df is not None:
                    # Merge with main data
                    merged_df = analyzer.merge_direction_info(
                        df, direction_df, 
                        'trajectory_with_direction.csv'
                    )
                    
                    if merged_df is not None:
                        # Analyze direction distribution
                        analyzer.analyze_direction_distribution(
                            merged_df, 
                            output_prefix='direction'
                        )
                        
                        # Analyze direction timing
                        analyzer.analyze_direction_timing(
                            merged_df, 
                            output_prefix='direction_timing'
                        )
            else:
                print("Target points file or Target column not found, skipping direction analysis")
                
        except Exception as e:
            print(f"Direction analysis skipped due to error: {e}")
        
        print("\n" + "=" * 50)
        print("Analysis completed successfully!")
        print(f"All results saved to: {analyzer.output_dir}")
        print("Generated files include:")
        print("- Substate distribution charts and statistics")
        print("- Interactive traffic maps (if location data available)")
        print("- Weekday vs Weekend comparisons (if time data available)")
        print("- Traffic violation analysis (if required columns available)")
        print("- Vehicle class distributions (if vehicle class data available)")
        print("- Direction and timing analysis (if target data available)")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please check your input data files and try again.")


if __name__ == "__main__":
    main()