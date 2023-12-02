# Import the necessary libraries
import os
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
        self.collisions = pd.DataFrame()
        self.collisions_order = None # Store the order of the aircrafts that are going to collide.
        self.precision = 2  # Set the precision for rounding coordinates, for collision detection.
        self.collision_count = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0} # Dictionary to store the number of collisions for each frame.
    

    def run_controller(self):
        print("# Running controller")

        # print(self.dataset_df.head())
        print("Number of dataset rows: " + str(len(self.dataset_df)))
        print("Number of unique aircrafts: " + str(len(self.dataset_df['AircraftID'].unique())))

        # TODO: Implement the algorithm here.
        # Step 1: Collision detection
        self.trajectory_collision_detection()

        # If there are no collisions, then we are done.
        if len(self.collisions) == 0:
            print("No collisions found")

        else:
            # Else, we move to the next step.
            # TODO: Step 2: Set order : Give priority to the aircrafts.
            self.set_order()

            # TODO: Step 3: Initialize airspace.
            # TODO: Step 4: While not convergent, do iterations.
            # TODO: Step 4.1: Calculate cost map.
            self.calculate_cost_map()

            # TODO: Step 4.2: Shortest path.
            # TODO: Step 4.3: Update trajectory.
            # TODO: Step 5: Issue Advisories to the aircrafts.

        print("# Controller finished running")
    

    """
    In this method, we will assign priorities to the aircrafts that are going to collide
    """
    def set_order(self):
        print("Setting order")

        # For each row of self.collision data the Frame is the point at which the aircrafts are going to collide,
        # so we take the AircraftID from this row of self.collision data, and 
        # we will find the lowest Frame number in the self.colliding_aircrafts_df. Then we will subtract the Frame number
        # of the self.collision data from the lowest Frame number in the self.colliding_aircrafts_df, and we will get the
        # number of frames that the aircrafts have to travel to reach the point of collision.
        # We will create a new column in the self.collision data called "Order" and we will store the number of frames
        # in this column.
        
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
        
        # Print the collisions_order DataFrame
        print(self.collisions_order)


    """
    This method calculates the cost map based on the collision risk of the aircrafts in the area
    """
    def calculate_cost_map(self):
        pass
    

    """
    This method will find the trajectories of the aircrafts that are going to collide, 
    Logic: If two aircrafts have the same coordinates in the same frame, then they are in collision.
    """
    def trajectory_collision_detection(self):

        # Plot trajectories
        # self.mark_trajectories(self.dataset_df)

        print("Finding collisions")
        # Group data for each unique combination of Frames
        grouped_data = self.dataset_df.groupby(['Frame'])

        # Iterate over each unique combination of Frames, and then
        for frame, group_df in grouped_data:
            # print(group_df)
            # Find aircrafts with the same x, y, and z coordinates within the same frame
            same_coordinates_df = self.find_same_coordinates(group_df)

            # Add the found collisions to the collisions DataFrame if there are any
            if len(same_coordinates_df) > 0:
                print(same_coordinates_df)
                self.collisions = pd.concat([self.collisions, same_coordinates_df], ignore_index=True)
                print("\n")

        # Print the number of collisions
        print(f"Total number of collisions: {sum(self.collision_count.values())}")
        for key, value in self.collision_count.items():
            if value > 0:
                print(f"{value} occurances of {key} aircrafts collisions.")
        
        # Get the full trajectory of the colliding aircrafts
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
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate (Height)')
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
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
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