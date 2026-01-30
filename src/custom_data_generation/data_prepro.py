# src/custom_data_generation/data_prepro.py

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Add the project root to the Python path to allow for absolute imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Correct for this file's location
sys.path.append(PROJECT_ROOT)

from src.config import DATA_PREPROCESSING_FALLBACK_SEED

# --- Configuration ---
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', '01_raw', 'synthetic_data')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', '02_processed')
SCALER_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'scalers')
SPLIT_RATIO = 0.2 # 20% for validation


def preprocess_synthetic_data(data_file_path=None, seed=None):
    """
    Loads raw synthetic datasets, performs an 80/20 train/validation split, standardizes
    the data based on the training set, and saves the processed data and scalers.

    Args:
        data_file_path (str, optional): A specific raw data file to preprocess.
                                     If None, all raw data will be processed.
        seed (int, optional): Random seed for train/validation split.
                              Defaults to value in config.
    """
    if seed is None:
        seed = DATA_PREPROCESSING_FALLBACK_SEED

    print("--- Starting Synthetic Data Preprocessing ---")
    print(f"Using random seed for split: {seed}")

    files_to_process = []
    if data_file_path:
        # If a specific file is targeted, ensure it's a CSV and exists
        if data_file_path.endswith('.csv') and os.path.exists(data_file_path):
            # Check if it's part of the synthetic data directory structure
            if os.path.abspath(data_file_path).startswith(os.path.abspath(RAW_DATA_DIR)):
                files_to_process.append(data_file_path)
    else:
        # If no target, walk through the synthetic raw data directory to find all CSV files
        for root, _, files in os.walk(RAW_DATA_DIR):
            for file in files:
                if file.endswith('.csv'):
                    files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        print("No synthetic data files found to process.")
        return None, None

    # This function is designed to process one file at a time as per the pipeline structure.
    raw_file_path = files_to_process[0]

    # --- 1. Setup Paths ---
    file = os.path.basename(raw_file_path)
    root = os.path.dirname(raw_file_path)
    subfolder = os.path.relpath(root, os.path.dirname(RAW_DATA_DIR))

    dataset_name = os.path.splitext(file)[0]
    processed_dataset_dir = os.path.join(PROCESSED_DATA_DIR, subfolder, dataset_name, str(seed))
    scaler_dataset_dir = os.path.join(SCALER_DIR, subfolder)
    os.makedirs(processed_dataset_dir, exist_ok=True)
    os.makedirs(scaler_dataset_dir, exist_ok=True)

    print(f"\nProcessing: {os.path.join(subfolder, file)}")

    # --- 2. Load and Split Data ---
    df = pd.read_csv(raw_file_path)
    X = df.drop('y', axis=1)
    y = df['y']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=SPLIT_RATIO, random_state=seed
    )

    # --- 3. Standardize Data ---
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

    X_val_scaled = x_scaler.transform(X_val)
    y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1))

    # --- 4. Save Scalers ---
    x_scaler_path = os.path.join(scaler_dataset_dir, f"{dataset_name}_seed-{seed}_x_scaler.joblib")
    y_scaler_path = os.path.join(scaler_dataset_dir, f"{dataset_name}_seed-{seed}_y_scaler.joblib")
    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(y_scaler, y_scaler_path)
    print(f"  - Saved X scaler to: {os.path.relpath(x_scaler_path, PROJECT_ROOT)}")
    print(f"  - Saved y scaler to: {os.path.relpath(y_scaler_path, PROJECT_ROOT)}")

    # --- 5. Save Processed Data ---
    train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    train_df['y'] = y_train_scaled.flatten()

    val_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    val_df['y'] = y_val_scaled.flatten()

    train_path = os.path.join(processed_dataset_dir, 'train.csv')
    val_path = os.path.join(processed_dataset_dir, 'validation.csv')
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print(f"  - Saved processed train/validation data to: {os.path.relpath(processed_dataset_dir, PROJECT_ROOT)}")

    print("\n--- Synthetic Data Preprocessing Complete ---")
    return train_path, val_path

if __name__ == "__main__":
    # This main block is for standalone testing and might need adjustments
    # since the function now processes one file and returns paths.
    # Example of how you might test it:
    test_file = os.path.join(RAW_DATA_DIR, 'output_noise', 'SYNTH_linear_2d_vars-1_samples-10000_noise-1.00.csv')
    if os.path.exists(test_file):
        train_p, val_p = preprocess_synthetic_data(data_file_path=test_file, seed=42)
        print(f"\nReturned paths:\nTrain: {train_p}\nValidation: {val_p}")
    else:
        print(f"Test file not found: {test_file}")