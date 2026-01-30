# An Approach to Enabling Standard Neural Networks to Accurately Model Equations for Engineering Problems Using IML

This project aims to demonstrate how to use Interpretable Machine Learning (IML) to enable standard neural networks to accurately model equations for engineering problems.

## Setup

### Prerequisites
*   Python 3.11.9 (available at https://www.python.org/downloads/release/python-3119/)

### Installation

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment:**
    *   On Windows:
        ```bash
        venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run the Pipeline
First, generate the data and train the models by running the main pipeline. This will create report files in `outputs/reports`.
```bash
python run_pipeline.py
```

### 2. Visualize Results
Use the interactive visualizer to generate plots from the reports.

1.  **Run the script:**
    ```bash
    python visualizer.py
    ```

2.  **Select a File:**
    The script will list all available CSV reports found in `outputs/reports`. Enter the number corresponding to the file you want to visualize.

3.  **Configure Plot:**
    *   If you selected an **ICE** file (1D or 2D), you will be asked: `Do you want a centered ICE plot? (y/n)`.
        *   Type `y` to center the curves/surfaces at the feature value closest to 0 (useful for seeing divergence relative to the origin).
        *   Type `n` for standard absolute predictions.
    *   If you selected a **2D-ICE** file, you will also be asked: `Do you want to plot a single instance? (y/n)`.
        *   Type `y` to visualize a single surface (less cluttered). You can specify the instance ID.
        *   Type `n` to visualize all surfaces.

4.  **Output:**
    The resulting images are saved in the `outputs/figures` directory.

## Utilities

### `src/utils.py`

This module contains utility functions for the project.

#### `get_short_model_name(dataset_name, model_details, seed)`

This function creates a shortened, standardized name for a model based on the dataset name, model architecture details, and the random seed used during training. This is useful for saving and identifying models.