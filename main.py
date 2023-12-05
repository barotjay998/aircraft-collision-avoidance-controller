from collision_avoidance_controller import DatasetReader, CollisionAvoidanceController

def main():

    # Specify the path to the dataset file
    dataset_path = 'dataset/combined_processed_data/' # OG dataset.
    # dataset_file_path = 'data_read_test_sample.txt'

    # Create an instance of DatasetReader
    dataset_reader = DatasetReader(dataset_path)

    # Read the data and get the DataFrame
    dataset_df = dataset_reader.read_dataset()

    if dataset_df is not None:
        # Create an instance of CollisionAvoidanceController
        controller = CollisionAvoidanceController(dataset_df)

        # Run the controller
        correction_df = controller.run_controller()
    
    # Save the correction DataFrame to a file
    correction_df.to_csv('output_correction.txt', sep=' ', index=False, header=False)


if __name__ == "__main__":
    main()