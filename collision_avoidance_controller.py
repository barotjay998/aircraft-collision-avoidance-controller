import pandas as pd

class DatasetReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_dataset(self):
        try:
            with open(self.file_path, 'r') as file:
                data = [line.strip().split() for line in file]
                
            columns = ["Frame", "AircraftID", "x", "y", "z", "windx", "windy"]
            df = pd.DataFrame(data, columns=columns, dtype=float)

            return df

        except Exception as e:
            print(f"Error reading dataset: {e}")
            return None
