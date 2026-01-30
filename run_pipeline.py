# run_pipeline.py

import os
import glob
import sys
import torch

# Add the project root to the Python path to allow for absolute imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.data_prepro import preprocess_data
from src.ann import SimpleNN, create_dataloaders, train_model, save_artifacts, get_model_details_str
from src.config import MASTER_RANDOM_SEEDS


def run_single_experiment(raw_data_path, config, seeds=None):
    """
    Runs the full pipeline for a single dataset.
    1. Preprocesses the data.
    2. Trains a baseline neural network.
    3. Saves the model and results.
    """
    dataset_name = os.path.splitext(os.path.basename(raw_data_path))[0]
    relative_path = os.path.relpath(os.path.dirname(raw_data_path), config['RAW_DATA_DIR'])

    print(f"\n{'='*20} RUNNING EXPERIMENT FOR: {dataset_name} {'='*20}")

    for i, seed in enumerate(seeds):
        print(f"\n{'-'*10} Running with seed {seed} ({i+1}/{len(seeds)} seeds) {'-'*10}")
        # --- Phase 1: Preprocessing ---
        print(f"\n--- Starting Phase 1: Preprocessing Data for {dataset_name} ---")
        
        # Creates processed data files and returns their paths
        train_path, val_path = preprocess_data(data_file_path=raw_data_path, seed=seed)

        if not train_path or not val_path:
            print(f"ERROR: Preprocessing failed for '{raw_data_path}'. Skipping training.")
            return


        # --- Phase 2: Baseline NN training ---
        print(f"\n--- Starting Phase 2: Training Baseline Model for {dataset_name} ---")

        # Set seed for reproducibility of model initialization and data shuffling
        torch.manual_seed(seed)
        
        # 1. Load and prepare data for Pytorch training
        train_loader, val_loader, input_size = create_dataloaders(
            train_path, val_path, config['BATCH_SIZE']
        )
        
        # 2. Initialize and train model
        model = SimpleNN(input_size=input_size, hidden_sizes=config['HIDDEN_SIZES'])

        trained_model, train_hist, val_hist, training_time = train_model(
            model, train_loader, val_loader, 
            config['EPOCHS'], config['LEARNING_RATE'], config['PATIENCE'], config['DEVICE']
        )

        # 3. Save artifacts (model and training history)
        save_path = os.path.join(config['BASELINE_MODELS_DIR'], relative_path)
        model_details = get_model_details_str(config['HIDDEN_SIZES'])
        save_artifacts(
            model=trained_model, 
            train_hist=train_hist, 
            val_hist=val_hist, 
            training_time=training_time,
            model_details=model_details,
            dataset_name=dataset_name, 
            save_path=save_path,
            seed=seed
        )
    print(f"--- Finished all training runs for {dataset_name} ---")


def main():
    """
    Orchestrates the entire project pipeline.
    Defines which datasets to run experiments on and executes the pipeline for each.
    """
    print("========================================")
    print("         STARTING FULL PIPELINE         ")
    print("========================================") 

    from pipeline_config import config
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
        seeds = MASTER_RANDOM_SEEDS.values()
        run_single_experiment(dataset_path, config, seeds=seeds)

    print("\n========================================")
    print("        FULL PIPELINE COMPLETED         ")
    print("========================================")

if __name__ == "__main__":
    # If no data is present, generate it
    # generate_all_data()
    
    main()