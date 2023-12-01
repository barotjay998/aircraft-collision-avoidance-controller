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

        print(self.dataset_df.head())

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
        # Plot the trajectories of each aircraft using x, y, and z coordinates
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        for aircraft_id, group in self.dataset_df.groupby('AircraftID'):
            ax.plot(group['x'], group['y'], group['z'], label=f'Aircraft {aircraft_id}')

        ax.set_xlabel('X Coordinate (km)')
        ax.set_ylabel('Y Coordinate (km)')
        ax.set_zlabel('Z Coordinate (km)')
        ax.set_title('Aircraft Trajectories')
        ax.legend()

        plt.show()