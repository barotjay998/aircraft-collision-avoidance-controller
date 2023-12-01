import pandas as pd


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
    def __init__(self, file_path):
        self.file_path = file_path

    def read_dataset(self):
        try:
            with open(self.file_path, 'r') as file:
                data = [line.strip().split() for line in file]

            # Data headers are based on information broadcased by the aircrafts with 
            # respect to the ADS-B data format specification.
            columns = ["Frame", "AircraftID", "x", "y", "z", "windx", "windy"]
            df = pd.DataFrame(data, columns=columns, dtype=float)

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

        print("Controller finished running.")