# Import the necessary libraries
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"""
Provides methods to read and preprocess the aircraft data.
"""
class DatasetReader:

    """
    Constructor
    Initializes the DatasetReader object, reads the processed dataset file of type txt

    :param file_path: The path to the processed dataset file.
    :type file_path: str
    """
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.dataframe_list = []

    def read_dataset(self):
        try:
            # Initialize an empty list to store DataFrames
            dataframes = []

            # Iterate over each file in the folder
            for file_name in os.listdir(self.folder_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(self.folder_path, file_name)
                    with open(file_path, 'r') as file:
                        data = [line.strip().split() for line in file]

                    # Create a DataFrame from the current file's data
                    columns = ["Frame", "AircraftID", "x", "y", "z", "windx", "windy"]
                    file_df = pd.DataFrame(data, columns=columns, dtype=float)

                    # Append the DataFrame to the list
                    dataframes.append(file_df)

            # Concatenate all DataFrames into a single DataFrame
            df = pd.concat(dataframes, ignore_index=True)

            return df

        except Exception as e:
            print(f"Error reading dataset: {e}")
            return None


class CollisionAvoidanceController:

    """
    Constructor
    Initializes the CollisionAvoidanceController object, reads the 
    data sent by the aircrafts in the area (ADS-B data), and stores it in a dataframe.

    :param dataset_df: The dataframe containing the data sent by the aircrafts in the area. 
    :type data: pandas.DataFrame
    """
    def __init__(self, dataset_df):
        self.dataset_df = dataset_df
        self.colliding_aircrafts_df = None # Store the trajectories of the colliding aircrafts.
        self.collisions = pd.DataFrame() # Store the collisions data, only consist of points of collision not the whole trajectory.
        self.collisions_order = None # Store the order of the aircrafts that are going to collide.
        # Set the precision for rounding coordinates, for collision detection.
        self.precision = 2  # The following precision is set to 10 meters, so that we can detect collisions.
        self.collision_count = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0} # Dictionary to store the number of collisions for each frame.
        self.iteration = 0 # Store the number of iterations the controller has run.
        self.convergence_time = 0
    

    def run_controller(self):
        print("# Running controller")

        # print(self.dataset_df.head())
        print("------------------------------------------------------------------------------")
        print("# ADS-B Controller Receiver Data Info:")
        print("------------------------------------------------------------------------------")
        print("Dataset rows: " + str(len(self.dataset_df)))
        print("Unique aircrafts: " + str(len(self.dataset_df['AircraftID'].unique())))
        print("------------------------------------------------------------------------------")
        print("# Controller Data Area Info:")
        print("------------------------------------------------------------------------------")
        print("Plane data range: Along the runway: " + str(self.dataset_df['x'].min()) + " to " + str(self.dataset_df['x'].max()) + " km")
        print("Plane data range: Perpendicular to the runway: " + str(self.dataset_df['y'].min()) + " to " + str(self.dataset_df['y'].max()) + " km")
        print("System height data range: " + str(self.dataset_df['z'].min()) + " to " + str(self.dataset_df['z'].max()) + " km")
        print("------------------------------------------------------------------------------")
        print("\n")
        
        # Run the controller untill there are no collisions or if it is the first iteration.
        while sum(self.collision_count.values()) > 0 or self.iteration == 0:

            if self.iteration > 0:
                # Reset the collision data, as we will now check for 
                # collisions in the updated trajectories.
                self.collisions = pd.DataFrame()
                self.collisions_order = None
                self.colliding_aircrafts_df = None
                self.collision_count = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

            # Step 1: Collision detection
            print("------------------------------------------------------------------------------")
            print(f"# Controller Iteration: {self.iteration + 1}")
            print("------------------------------------------------------------------------------")
            self.trajectory_collision_detection()

            # If there are no collisions, then we are done.
            if sum(self.collision_count.values()) == 0:
                print("------------------------------------------------------------------------------")
                print("# No more collisions found: Course correction completed")
                print("------------------------------------------------------------------------------")
                break

            else:
                # Step 2: Set order: Give priority to the colliding aircrafts.
                self.set_order()

                # Step 3: Course correction: Correct the trajectories of the colliding aircrafts.
                self.course_correction()

                # Step 4: Update the dataset
                self.update_trajectory()

        print("# Controller finished running")
        
    """
    This method updates the dataset to contain only the data about the course corrected aircrafts,
    for the next iteration and checking for collisions.
    """
    def update_trajectory(self):
        print("# Updating trajectory data...")
        print("------------------------------------------------------------------------------")
        print("\n")

        self.dataset_df = self.colliding_aircrafts_df

        # Update the iteration number
        self.iteration += 1
    
    """
    In this method, we will assign priorities to the aircrafts that are going to collide
    # Determine the number of frames aircrafts have to travel to reach the collision point:
    # 1. Extract AircraftID from each row in self.collision data.
    # 2. Find the lowest Frame number in self.colliding_aircrafts_df.
    # 3. Subtract the Frame number of self.collision data from the lowest Frame number.
    # 4. Create a new dataframe with "Order" column similar to self.collision data.
    """
    def set_order(self):
        print("------------------------------------------------------------------------------")
        print("# Setting priority order...")
        
        # Check if collisions DataFrame is not empty
        if not self.collisions.empty:
            # Create a new DataFrame to store collisions with 'Order' column
            self.collisions_order = self.collisions.copy()

            for index, collision_row in self.collisions_order.iterrows():
                # Get AircraftID and Frame from the collision row
                aircraft_id = collision_row['AircraftID']
                collision_frame = collision_row['Frame']

                # Filter colliding_aircrafts_df for the current AircraftID
                aircraft_group = self.colliding_aircrafts_df[self.colliding_aircrafts_df['AircraftID'] == aircraft_id]

                # Find the lowest Frame number in colliding_aircrafts_df
                lowest_frame = aircraft_group['Frame'].min()

                # Calculate the number of frames to reach the point of collision
                frames_to_collision = collision_frame - lowest_frame

                # Update the 'Order' column in collisions_order DataFrame
                self.collisions_order.loc[index, 'Order'] = frames_to_collision
        
        self.collisions_order = self.collisions_order.sort_values(by='Order')


    """
    This method calculates the cost map based on the collision risk of the aircrafts in the area
    The approach involves updating the z-coordinate in the colliding_aircrafts_df's collision frame gradually from five 
    frames before the collision. Starting from there, we incrementally adjust the z-coordinate by for example +0.0914 per frame. 
    This ensures that by the collision frame, the z-coordinate reaches the advised Climb Level of +0.457. 
    All these adjustments are made within the colliding_aircrafts_df DataFrame, as for this iteration only these trajectories 
    need a course correction.
    """
    def course_correction(self):
        print("------------------------------------------------------------------------------")
        print("# Trajectory correction...")
        print("------------------------------------------------------------------------------")

        # Trajectory correction:
        # Is based on vertical rate manipulation, we adjust our actions base on TCAS climb advisories SCL2500, CL2000, CL1500
        # CL1000, CL500 (extra). In the vertical_rate_of_change we convert these in to km metric from ft metric, and assign
        # a gradual rate of change as value.
        vertical_rate_of_change = {0.7620: 0.1524, 0.6096: 0.12192, 0.4572: 0.0914, 0.3048: 0.06096, 0.1524: 0.03048}

        # Used to make sure that consecutive aircrafts have different vertical rates,
        # we initialize it with a SCL2500 value so that the first aircraft avoids 
        # Strict Climb advisory (SCL).
        antecedent_vertical_rate, vertical_rate = 0.7620, 0.7620

        # We go through each aircraft which is colliding in order of priority assigned
        # in the collisions_order DataFrame.
        for index, collision_row in self.collisions_order.iterrows():
            # Get AircraftID and Frame at which the collision is going to happen.
            aircraft_id = collision_row['AircraftID']
            collision_frame = collision_row['Frame']

            # Filter colliding_aircrafts_df for the current AircraftID
            aircraft_group = self.colliding_aircrafts_df[self.colliding_aircrafts_df['AircraftID'] == aircraft_id]

            # Find the index for the collision frame in the colliding_aircrafts_df
            collision_index = aircraft_group[aircraft_group['Frame'] == collision_frame].index
            
            # We to the course correction for each of the collision index of the colliding aircraft.
            # This loop ensures course correction for multiple collisions of the same aircraft along its trajectory.
            for collision_idx in collision_index:

                # collision_index = collision_index.item()
                # print(f"Collision index: {collision_idx} for AircraftID: {aircraft_id}")

                # Change vertical rate
                while vertical_rate == antecedent_vertical_rate:
                    vertical_rate = random.choice(list(vertical_rate_of_change.keys()))
                vertical_rate_multiplier = vertical_rate_of_change[vertical_rate]

                # This is used to update the z-coordinate gradually from five frames before 
                # the collision frame to the collision frame.
                multiplier = 6

                # Iterate from five frames before the collision frame to the collision frame
                for i in range(collision_idx - 5, collision_idx + 1):
                    # Gradually update the z-coordinate
                    multiplier -= 1
                    update_value = vertical_rate - vertical_rate_multiplier * (multiplier)
                    self.colliding_aircrafts_df.loc[i, 'z'] += update_value
                    # print(f"Updating the z-coordinate for index {i} to + {update_value}")

                # Update antecedent_vertical_rate
                antecedent_vertical_rate = vertical_rate
    

    """
    This method will find the trajectories of the aircrafts that are going to collide, 
    Logic: If two aircrafts have the same coordinates in the same frame, then they are in collision.
    """
    def trajectory_collision_detection(self):

        # Plot trajectories
        # self.mark_trajectories(self.dataset_df)

        # Group data for each unique combination of Frames
        grouped_data = self.dataset_df.groupby(['Frame'])

        print("------------------------------------------------------------------------------")
        print("# Monitoring Collisions")
        print("------------------------------------------------------------------------------")
        # Iterate over each unique combination of Frames, and then
        for frame, group_df in grouped_data:
            # print(group_df)
            # Find aircrafts with the same x, y, and z coordinates within the same frame
            same_coordinates_df = self.find_same_coordinates(group_df)
            # Add the found collisions to the collisions DataFrame if there are any
            if len(same_coordinates_df) > 0:
                print(f"Collision # {sum(self.collision_count.values())}")
                print(same_coordinates_df)
                self.collisions = pd.concat([self.collisions, same_coordinates_df], ignore_index=True)
                print("------------------------------------------------------------------------------")

        if not self.collisions.empty:
            print("# Collision Monitor Summary")
            print("------------------------------------------------------------------------------")
            # Print the number of collisions
            print(f"Total number of collisions: {sum(self.collision_count.values())}")
            for key, value in self.collision_count.items():
                if value > 0:
                    print(f"{value} occurances of {key} aircrafts collisions.")
            print("------------------------------------------------------------------------------")
            print("\n")
            
            # Get the full trajectory of the colliding aircrafts if there are any.
            # We do that by creating a DataFrame with only rows for colliding aircrafts.
            self.colliding_aircrafts_df = self.dataset_df[self.dataset_df['AircraftID'].isin(self.collisions['AircraftID'])]

        # Export colliding_aircrafts_df to a text file
        # self.colliding_aircrafts_df.to_csv('colliding_aircrafts.txt', sep=' ', index=False)

        # Plot trajectories
        # self.mark_trajectories(self.colliding_aircrafts_df)


    """
    This method takes a group of aircrafts that exist in the same frame, and finds out which of them
    have thet same coordinates around some precision, and returns a dataframe containing the aircrafts.

    :param group_df: The dataframe containing the aircrafts in the same frame.
    :type group_df: pandas.DataFrame
    """
    def find_same_coordinates(self, group_df):
        # Round the coordinates to the specified precision for checking duplicates
        group_df_rounded = group_df.round({'x': self.precision, 'y': self.precision, 'z': self.precision})

        # Find rows with the same rounded coordinates within the group
        same_coordinates_df = group_df[group_df_rounded.duplicated(['x', 'y', 'z'], keep=False)]
        cord_df_modified = group_df[group_df_rounded.duplicated(['x', 'y', 'z'], keep=False)]
        #modify the df
        # Drop duplicate rows based on AircraftID
        # LOGIC: If there are same coordinates for same aircraft, then it is not a collision, the
        # aircraft is just hovering in the same place.
        same_coordinates_df = same_coordinates_df.drop_duplicates(subset=['AircraftID'])

        # Update collison count, if we found any collisions.
        if len(same_coordinates_df) > 1:
            # We update the collsion count based on the number of aircrafts in collision.
            self.collision_count[len(same_coordinates_df)] += 1
            return same_coordinates_df

        else:
            # No collisions found
            return pd.DataFrame()
    

    """
    This method marks the trajectories of the aircrafts in the area, 
    by plotting them in a 3D plot and a 2D plot.
    """
    def mark_trajectories(self, trajectory_df):
        print("Plotting trajectories")
        self.plot_trajectories_3d(trajectory_df)
        self.plot_trajectories_2d(trajectory_df)


    """
    This method plots the trajectories of the aircrafts in the area in a 3D plot.
    """
    def plot_trajectories_3d(self, trajectory_df):
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Iterate over each unique AircraftID
        for aircraft_id in trajectory_df['AircraftID'].unique():
            # Filter data for the current aircraft
            aircraft_data = trajectory_df[trajectory_df['AircraftID'] == aircraft_id]

            # Plot trajectory
            ax.plot(aircraft_data['x'], aircraft_data['y'], aircraft_data['z'], label=f'Aircraft {int(aircraft_id)}')

        # Set labels and title
        ax.set_xlabel('X (in km)')
        ax.set_ylabel('Y (in km)')
        ax.set_zlabel('Z (in km) (Height)')
        ax.set_title('Aircraft Trajectories')

        # Get the maximum absolute values for x and y coordinates
        max_abs_x = max(abs(coord) for coord in ax.get_xlim())
        max_abs_y = max(abs(coord) for coord in ax.get_ylim())

        # Set the limits based on the maximum absolute values
        ax.set_xlim(-max_abs_x, max_abs_x)
        ax.set_ylim(-max_abs_y, max_abs_y)

        # Show legend
        ax.legend()

        # Show the plot
        plt.show()
    

    """
    This method plots the trajectories of the aircrafts in the area in a 2D plot.
    """
    def plot_trajectories_2d(self, trajectory_df):
        # Create a 2D plot
        ax = plt.subplot(122)

        # Iterate over each unique AircraftID
        for aircraft_id in trajectory_df['AircraftID'].unique():
            # Filter data for the current aircraft
            aircraft_data = trajectory_df[trajectory_df['AircraftID'] == aircraft_id]

            # Scatter plot with color representing height
            scatter = ax.scatter(
                aircraft_data['x'],
                aircraft_data['y'],
                c=aircraft_data['z'],
                cmap='YlOrRd',  # Choose the colormap (Yellow to Orange to Red)
                label=f'Aircraft {int(aircraft_id)}',
                linewidth=0.5
            )

        # Set labels and title
        ax.set_xlabel('X (in km)')
        ax.set_ylabel('Y (in km)')
        ax.set_title('Aircraft Trajectories in 2D with Z-coordinate (Height) Color')

        max_abs_x = max(abs(coord) for coord in ax.get_xlim())
        max_abs_y = max(abs(coord) for coord in ax.get_ylim())

        # Set x=0, y=0 at the center of the x, y plane
        # ax.set_xlim(-max_abs_x, max_abs_x)
        # ax.set_ylim(-max_abs_y, max_abs_y)

        ax.set_xlim(-6, 6)  # Adjust as needed
        ax.set_ylim(-6, 6)  # Adjust as needed
        ax.set_aspect('equal')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Z Coordinate (Height)')

        # Show the plot
        plt.show()

    
    """Shortest path for converging"""
    def dijkstra(self, src):
 
        dist = [1e7] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
 
        for cout in range(self.V):
 
            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minDistance(dist, sptSet)
 
            # Put the minimum distance vertex in the
            # shortest path tree
            sptSet[u] = True
 
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):
                if (self.graph[u][v] > 0 and
                   sptSet[v] == False and
                   dist[v] > dist[u] + self.graph[u][v]):
                    dist[v] = dist[u] + self.graph[u][v]
 
        return sptSet

    def get_paths(self):
        paths = self.dijkstra(1)
        for path in paths:
            if path:
                print("no collision on this path")
        
        return
