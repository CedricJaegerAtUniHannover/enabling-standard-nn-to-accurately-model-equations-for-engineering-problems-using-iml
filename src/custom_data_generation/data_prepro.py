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

from src.config import DATA_PREPROCESSING_SEED

# --- Configuration ---
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', '01_raw', 'synthetic_data')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', '02_processed')
SCALER_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'scalers')
SPLIT_RATIO = 0.2 # 20% for validation
SPLIT_SEED = DATA_PREPROCESSING_SEED

def preprocess_synthetic_data():
    """
    Loads raw synthetic datasets, performs an 80/20 train/validation split, standardizes
    the data based on the training set, and saves the processed data and scalers.
    """
    print("--- Starting Synthetic Data Preprocessing ---")
    print(f"Using random seed for split: {SPLIT_SEED}")

    # Walk through the synthetic raw data directory to find all CSV files
    for root, _, files in os.walk(RAW_DATA_DIR):
        for file in files:
            if not file.endswith('.csv'):
                continue

            # --- 1. Setup Paths ---
            raw_file_path = os.path.join(root, file)
            # Get the subfolder structure relative to the parent of RAW_DATA_DIR
            # to replicate it in the processed and artifacts directories.
            subfolder = os.path.relpath(root, os.path.dirname(RAW_DATA_DIR))

            # Create specific output directories for this dataset
            dataset_name = os.path.splitext(file)[0]
            processed_dataset_dir = os.path.join(PROCESSED_DATA_DIR, subfolder, dataset_name)
            scaler_dataset_dir = os.path.join(SCALER_DIR, subfolder)
            os.makedirs(processed_dataset_dir, exist_ok=True)
            os.makedirs(scaler_dataset_dir, exist_ok=True)

            print(f"\nProcessing: {os.path.join(subfolder, file)}")

            # --- 2. Load and Split Data ---
            df = pd.read_csv(raw_file_path)
            X = df.drop('y', axis=1)
            y = df['y']

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=SPLIT_RATIO, random_state=SPLIT_SEED
            )

            # --- 3. Standardize Data ---
            # Initialize scalers
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()

            # Fit scalers ONLY on the training data
            X_train_scaled = x_scaler.fit_transform(X_train)
            # Reshape y for the scaler, which expects a 2D array
            y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

            # Transform the validation data using the fitted scalers
            X_val_scaled = x_scaler.transform(X_val)
            y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1))

            # --- 4. Save Scalers ---
            x_scaler_path = os.path.join(scaler_dataset_dir, f"{dataset_name}_x_scaler.joblib")
            y_scaler_path = os.path.join(scaler_dataset_dir, f"{dataset_name}_y_scaler.joblib")
            joblib.dump(x_scaler, x_scaler_path)
            joblib.dump(y_scaler, y_scaler_path)
            print(f"  - Saved X scaler to: {os.path.relpath(x_scaler_path, PROJECT_ROOT)}")
            print(f"  - Saved y scaler to: {os.path.relpath(y_scaler_path, PROJECT_ROOT)}")

            # --- 5. Save Processed Data ---
            train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            train_df['y'] = y_train_scaled.flatten()

            val_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
            val_df['y'] = y_val_scaled.flatten()

            train_df.to_csv(os.path.join(processed_dataset_dir, 'train.csv'), index=False)
            val_df.to_csv(os.path.join(processed_dataset_dir, 'validation.csv'), index=False)
            print(f"  - Saved processed train/validation data to: {os.path.relpath(processed_dataset_dir, PROJECT_ROOT)}")

    print("\n--- Synthetic Data Preprocessing Complete ---")

if __name__ == "__main__":
    preprocess_synthetic_data()