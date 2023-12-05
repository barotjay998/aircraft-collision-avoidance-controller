# Aircraft Collision Avoidance System

Aircraft Collision Avoidance System is an advanced Python implementation for the real-time avoidance of collisions among multiple aircraft, employing an autonomous dependent surveillance-broadcast (ADS-B) system. The core aim is to utilize satellite-based navigation technology to address identified shortcomings in the existing Traffic Collision Avoidance System (TCAS). The system is designed to process trajectory data, identify potential collisions, and generate advisories to prevent accidents in air travel. 

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Dataset](#dataset)
  - [Running the Code](#running-the-code)
  - [Monitoring the Controller](#monitoring-the-controller)
- [Testing](#testing)
- [Visualization](#visualization)
- [License](#license)

## Overview

Aircraft Collision Avoidance System is designed to process trajectory data, identify potential collisions, and generate advisories to prevent accidents in air travel.

## Getting Started

### Prerequisites

Before running the Aircraft Collision Avoidance System, ensure that you have the following prerequisites installed:

- Python 3.x
- Pip (Python package installer)

### Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/barotjay998/aircraft-collision-avoidance-controller.git
    ```

2. Navigate to the project directory:

    ```bash
    cd aircraft-collision-avoidance-contoller
    ```

3. Install the required dependencies using the provided `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Dataset

To use the Aircraft Collision Avoidance System with your own ADS-B formatted trajectory data, follow these steps:

1. Obtain ADS-B formatted trajectory data or download a sample dataset from [TrajAir](https://theairlab.org/trajair/).

2. Place the dataset files in a folder of your choice.

### Running the Code

1. Open the `main.py` file in a text editor.

2. Specify the path to your dataset folder:

    ```python
    dataset_path = "path/to/your/dataset"
    ```

3. Save the `main.py` file.

4. Execute the `main.py` file:

    ```bash
    python main.py
    ```

### Monitoring the Controller

The controller's execution can be monitored on the command line. Once the execution is complete, an advisory file will be generated.

## Testing

For testing purposes, two sample dataset files are provided:

- `2_colliding_aircrafts_data.txt`: An instance where the collision of two aircraft is captured.
- `7d1_colliding_aircrafts_data.txt`: Data collected over a week containing multiple possible collisions.

## Visualization

To visualize trajectories at various stages of the algorithm, use the `mark_trajectories()` method of the controller. This can provide useful insights into the collision avoidance process.

## License

This project is licensed under the [MIT License](LICENSE).