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
    
    def run_controller(self):
        print("Running controller...")

        # print(self.dataset_df.head())

        # TODO: Implement the algorithm here.
        #
        # TODO: Step 1: COLLISIONDETECTION - Collision Risk-Based Cost Map.
        self.calculate_cost_map()

        print("Controller finished running.")
    

    """
    This method calculates the cost map based on the collision risk of the aircrafts in the area,
    the first step to do this is to mark the trajectories of each aircraft in the area, based upon the 
    points of traversal of each aircraft.
    """
    def calculate_cost_map(self):
    
        self.mark_trajectories()
    
    
    def mark_trajectories(self):
        self.plot_trajectories_3d()
        self.plot_trajectories_2d()


    def plot_trajectories_3d(self):
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Iterate over each unique AircraftID
        for aircraft_id in self.dataset_df['AircraftID'].unique():
            # Filter data for the current aircraft
            aircraft_data = self.dataset_df[self.dataset_df['AircraftID'] == aircraft_id]

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
    

    def plot_trajectories_2d(self):
        # Create a 2D plot
        ax = plt.subplot(122)

        # Iterate over each unique AircraftID
        for aircraft_id in self.dataset_df['AircraftID'].unique():
            # Filter data for the current aircraft
            aircraft_data = self.dataset_df[self.dataset_df['AircraftID'] == aircraft_id]

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