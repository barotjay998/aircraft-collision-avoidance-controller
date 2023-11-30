from collision_avoidance_controller import DatasetReader

def main():

    # Reading the dataset and storing it in a dataframe.

    # dataset_file_path = 'dataset/7days1/processed_data/test/2.txt' # OG dataset.
    dataset_file_path = 'data_read_test_sample.txt'
    dataset_reader = DatasetReader(dataset_file_path)
    dataset_df = dataset_reader.read_dataset()

    if dataset_df is not None:
        print("Dataset read successfully:")
        print(dataset_df.head())

if __name__ == "__main__":
    main()