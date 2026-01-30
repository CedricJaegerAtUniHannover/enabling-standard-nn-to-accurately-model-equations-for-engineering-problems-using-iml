# run_pipeline.py

import os
import glob
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add the project root to the Python path to allow for absolute imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.data_prepro import preprocess_data
from src.ann import SimpleNN, train_model, save_model_and_history, get_model_details_str
from src.config import MODEL_TRAINING_SEEDS


def run_single_experiment(raw_data_path, config):
    """
    Runs the full pipeline for a single dataset.
    1. Preprocesses the data.
    2. Trains a baseline neural network.
    3. Saves the model and results.
    """
    print(f"\n{'='*20} RUNNING EXPERIMENT FOR: {os.path.basename(raw_data_path)} {'='*20}")

    # --- Phase 1: Preprocessing ---
    # This will create the processed files in data/02_processed/
    preprocess_data(target_file=raw_data_path)

    # --- Determine paths for processed data and model saving ---
    # e.g., 'data/01_raw/synthetic_data/input_noise/FILE.csv' -> 'synthetic_data/input_noise'
    relative_path = os.path.relpath(os.path.dirname(raw_data_path), config['RAW_DATA_DIR'])
    
    # The preprocessor creates a directory named after the raw file, containing train/val splits
    dataset_name = os.path.splitext(os.path.basename(raw_data_path))[0]
    processed_dataset_dir = os.path.join(config['PROCESSED_DATA_DIR'], relative_path, dataset_name)

    train_path = os.path.join(processed_dataset_dir, 'train.csv')
    val_path = os.path.join(processed_dataset_dir, 'validation.csv')

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"ERROR: Processed train/validation files not found in '{processed_dataset_dir}'. Skipping training.")
        return

    # --- Phase 2: NN Training ---
    print(f"\n--- Starting Phase 2: Training Baseline Model for {dataset_name} ---")
    
    # 1. Load and Prepare Data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df.drop('y', axis=1).values
    y_train = train_df['y'].values
    X_val = val_df.drop('y', axis=1).values
    y_val = val_df['y'].values

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # The preprocessor already created the train/val split, so we use them directly
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'])

    # Loop over the seeds to train multiple models for the same dataset
    for model_run_name, seed in MODEL_TRAINING_SEEDS.items():
        print(f"\n--- Training Run: {model_run_name} (Seed: {seed}) ---")

        # Set seed for reproducibility of model initialization and data shuffling
        torch.manual_seed(seed)

        # 2. Initialize and Train Model
        input_size = X_train.shape[1]
        model = SimpleNN(input_size=input_size, hidden_sizes=config['HIDDEN_SIZES'])

        trained_model, train_hist, val_hist = train_model(
            model, train_loader, val_loader, 
            config['EPOCHS'], config['LEARNING_RATE'], config['PATIENCE'], config['DEVICE']
        )

        # 3. Save Artifacts
        model_details = get_model_details_str(config['HIDDEN_SIZES'])
        run_number = model_run_name.split('_')[-1]
        model_name = f"{dataset_name}_ann_{model_details}_run-{run_number}_seed-{seed}_status-best"

        save_path = os.path.join(config['BASELINE_MODELS_DIR'], relative_path)
        
        history = {"train_loss": train_hist, "validation_loss": val_hist}
        save_model_and_history(trained_model, history, save_path, model_name)
    print(f"--- Finished all training runs for {dataset_name} ---")


def main():
    """
    Orchestrates the entire project pipeline.
    Defines which datasets to run experiments on and executes the pipeline for each.
    """
    print("========================================")
    print("         STARTING FULL PIPELINE         ")
    print("========================================") 

    # --- Pipeline Configuration ---
    config = {
        "RAW_DATA_DIR": "data/01_raw",
        "PROCESSED_DATA_DIR": "data/02_processed",
        "BASELINE_MODELS_DIR": "models/01_baseline",
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # Model Hyperparameters
        "HIDDEN_SIZES": [64, 64, 64],
        "EPOCHS": 200,
        "LEARNING_RATE": 0.001,
        "PATIENCE": 15,
        "BATCH_SIZE": 32
    }
    print(f"Using device: {config['DEVICE']}")

    # --- Define which datasets to process ---
    search_path = os.path.join(config['RAW_DATA_DIR'], "**", "*.csv")
    datasets_to_process = glob.glob(search_path, recursive=True)

    if not datasets_to_process:
        print("No raw data files found to process. Exiting.")
        return

    print(f"\nFound {len(datasets_to_process)} datasets to process.")

    # --- Run the pipeline for each dataset ---
    for dataset_path in datasets_to_process:
        run_single_experiment(dataset_path, config)

    print("\n========================================")
    print("        FULL PIPELINE COMPLETED         ")
    print("========================================")

if __name__ == "__main__":
    # If no data is present, generate it
    # generate_all_data()
    
    main()