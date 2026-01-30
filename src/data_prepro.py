# src/data_prepro.py

import os
import sys

# Add the project root to the Python path to allow for absolute imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# Import the specific preprocessing function for synthetic data
from src.custom_data_generation.data_prepro import preprocess_synthetic_data
from src.config import RAW_DATA_DIR

def preprocess_data(data_file_path, seed=None):
    """
    Preprocesses a single data file.

    If the file is identified as a synthetic dataset, it calls the specific
    synthetic data preprocessing function. Otherwise, it's a placeholder for other
    data types.

    Args:
        data_file_path (str): The full path to the raw data file.
    """
    # Determine if the file is a custom synthetic dataset
    if 'synthetic_data' in data_file_path:
        print(f"--- Dispatching to synthetic data preprocessor for: {os.path.basename(data_file_path)} ---")
        return preprocess_synthetic_data(data_file_path=data_file_path, seed=seed)
    else:
        # TODO: Implement preprocessing for other data sources.
        # This section would handle any non-synthetic datasets.
        print(f"--- Skipping non-synthetic file: {os.path.basename(data_file_path)} (no processor defined) ---")
        return None, None

def preprocess_all_data():
    """
    Iterates through all files in the raw data directory (and its subdirectories)
    and calls the appropriate preprocessor for each file.
    """
    print("--- Starting Full Data Preprocessing ---")
    
    # Walk through the entire raw data directory
    for root, _, files in os.walk(RAW_DATA_DIR):
        for file in files:
            # Only CSV files are recognized as data files
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                preprocess_data(file_path)

    print("\n--- All Data Preprocessing Complete ---")

if __name__ == "__main__":
    # When run directly, process all data.
    preprocess_all_data()
