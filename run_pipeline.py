# run_pipeline.py

import os
import glob
import sys
import warnings

# Suppress juliacall warning about torch import order.
# On Windows torch has to be imported first, which juliapkg from pysr dislikes, to avoid DLL initialization errors (WinError 1114).
warnings.filterwarnings("ignore", message=".*torch was imported before juliacall.*")

import torch
import numpy as np
import pandas as pd
import random

# Add the project root to the Python path to allow for absolute imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.ice import get_ice_curves, get_ice_surfaces
from src.h_statistic import get_friedman_h_statistic
from src.utils import get_short_model_name
from src.data_prepro import load_scalers, unstandardize_ice_data, preprocess_data
from src.data_augmentation import augment_dataset
from src.ann import SimpleNN, create_dataloaders, train_model, save_artifacts, get_model_details_str
from src.config import MASTER_RANDOM_SEEDS
from pipeline_config import config
from src.custom_data_generation.generator import generate_data as generate_all_data

# Wrapper class for PyTorch model to be compatible with iML libraries
class PyTorchModelWrapper:
    """
    Wraps a PyTorch model to provide a scikit-learn-like .predict() method
    that accepts a pandas DataFrame or NumPy array.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, X):
        """
        Makes predictions with the PyTorch model.

        Args:
            X (pd.DataFrame or np.ndarray): Input data.

        Returns:
            np.ndarray: Model predictions.
        """
        if isinstance(X, pd.DataFrame):
            X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        elif isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            raise TypeError(f"Unsupported input type for prediction: {type(X)}")
            
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy().flatten()

def set_global_seeds(seed):
    """
    Sets seeds for various libraries to ensure reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def perform_1d_ice_analysis(model_wrapper, X_ice, ice_grid_res, report_path, model_name, dataset_name, seed, relative_path):
    """Calculates and saves 1D-ICE curves."""
    print("Calculating 1D-ICE curves for all features...")
    all_ice_curves = []
    for feature in X_ice.columns:
        # Use 2x resolution for 1D curves as they are cheaper than 2D surfaces
        ice_df = get_ice_curves(model_wrapper, X_ice, feature, centered=False, num_grid_points=ice_grid_res*2)
        all_ice_curves.append(ice_df)
    
    # Save standardized 1D-ICE results
    if all_ice_curves:
        full_ice_df = pd.concat(all_ice_curves, ignore_index=True)
        ice_file_path = os.path.join(report_path, f"stand_1d-ICE_{model_name}.csv")
        full_ice_df.to_csv(ice_file_path, index=False)
        print(f"Saved standardized 1D-ICE curves to {ice_file_path}")

        # --- Un-standardization Step ---
        # The relative_path is from the raw data directory, which mirrors the scaler artifact structure
        x_scaler, y_scaler = load_scalers(dataset_name, seed, relative_path)
        unstandardized_ice_1d_df = unstandardize_ice_data(full_ice_df, x_scaler, y_scaler, relative_path)
        unstd_ice_path = os.path.join(report_path, f"unstand_1d-ICE_{model_name}.csv")
        unstandardized_ice_1d_df.to_csv(unstd_ice_path, index=False)
        print(f"Saved un-standardized 1D-ICE curves to {unstd_ice_path}")

def perform_interaction_analysis(model_wrapper, X_train, X_ice, ice_num_samples, ice_grid_res, report_path, model_name, dataset_name, seed, relative_path, h_stat_threshold):
    """Calculates H-statistics and 2D-ICE surfaces for interacting features."""
    # Friedman H-statistic for feature interaction ranking
    print("Calculating Friedman's pairwise H-statistic for feature interactions...")
    _, pairwise_h_stats = get_friedman_h_statistic(model_wrapper, X_train, sample_size=ice_num_samples, random_state=seed)
    
    # Save pairwise H-statistic results
    pairwise_h_path = os.path.join(report_path, f"h-statistic_pairwise_{model_name}.csv")
    pairwise_h_stats.to_csv(pairwise_h_path)
    print(f"Saved pairwise H-statistic results to {pairwise_h_path}")

    # 2d-ICE curves for interacting feature pairs based on H-statistic
    print("Calculating 2D-ICE surfaces for significant interacting features...")
    
    stacked_h = pairwise_h_stats.stack()
    stacked_h.index = stacked_h.index.map(lambda x: tuple(sorted(x)))
    stacked_h = stacked_h.drop_duplicates()
    stacked_h = stacked_h[stacked_h.index.get_level_values(0) != stacked_h.index.get_level_values(1)]
    
    significant_pairs_series = stacked_h[stacked_h > h_stat_threshold].sort_values(ascending=False)
    top_pairs = significant_pairs_series.index.tolist()

    all_ice_surfaces = []
    if top_pairs:
        print(f"Found {len(top_pairs)} pairs with H-statistic > {h_stat_threshold} to analyze: {top_pairs}")
        for pair in top_pairs:
            print(f"  - Calculating 2D-ICE for pair: {pair}")
            ice2d_df = get_ice_surfaces(model_wrapper, X_ice, features=list(pair), centered=False, num_grid_points=ice_grid_res)
            all_ice_surfaces.append(ice2d_df)

    # Save standardized 2D-ICE results
    if all_ice_surfaces:
        full_ice2d_df = pd.concat(all_ice_surfaces, ignore_index=True)
        ice2d_file_path = os.path.join(report_path, f"stand_2d-ICE_{model_name}.csv")
        full_ice2d_df.to_csv(ice2d_file_path, index=False)
        print(f"Saved standardized 2D-ICE surfaces to {ice2d_file_path}")

        # --- Un-standardization Step ---
        x_scaler, y_scaler = load_scalers(dataset_name, seed, relative_path)
        unstandardized_ice_2d_df = unstandardize_ice_data(full_ice2d_df, x_scaler, y_scaler, relative_path)
        unstd_ice2d_path = os.path.join(report_path, f"unstand_2d-ICE_{model_name}.csv")
        unstandardized_ice_2d_df.to_csv(unstd_ice2d_path, index=False)
        print(f"Saved un-standardized 2D-ICE surfaces to {unstd_ice2d_path}")
    else:
        print("No significant feature pairs found for 2D-ICE analysis.")

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
        
        # Set global seeds for reproducibility at the start of the iteration
        set_global_seeds(seed)

        # --- Phase 1: Preprocessing ---
        print(f"\n--- Starting Phase 1: Preprocessing Data for {dataset_name} ---")
        
        # Creates processed data files and returns their paths
        train_path, val_path = preprocess_data(data_file_path=raw_data_path, seed=seed)

        if not train_path or not val_path:
            print(f"ERROR: Preprocessing failed for '{raw_data_path}'. Skipping training.")
            return


        # --- Phase 2: Baseline NN training ---
        print(f"\n--- Starting Phase 2: Training Baseline Model for {dataset_name} ---")

        # Re-set seed for reproducibility of model initialization and data shuffling
        # This ensures that any potential (overlooked) random state changes in Phase 1 do not affect Phase 2
        set_global_seeds(seed)
        
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

        # TODO: loop of log2[#number of features] augmentations and retrainings because only ever two features of 
        # the previous iteration can be augmented together thus there are so many iterations that a
        # function terms with all features in them could be found (NOTE: this is a rough approximation
        # and simplicification that would not hold true if features would have more complex operations attached to them)
        iteration = 0
        report_path = os.path.join(config['REPORTS_DIR'], relative_path, f"iteration_{iteration}")
        
        # --- Phase 3: Interpreting the Model for Feature Behaviour and Interactions ---
        print("\n--- Starting Phase 3: Interpreting the Model ---")

        # Re-set seed before interpretation to ensure consistent sampling
        set_global_seeds(seed)

        # Create a wrapper for the PyTorch model to make it compatible with iML libraries
        model_wrapper = PyTorchModelWrapper(trained_model, config['DEVICE'])
        
        # Load the training data for iML analysis
        train_df = pd.read_csv(train_path)
        X_train = train_df.drop('y', axis=1)
        
        # Ensure the report directory exists
        os.makedirs(report_path, exist_ok=True)
        
        # Get a consistent model name for report files
        model_name = get_short_model_name(dataset_name, model_details, seed)

        # --- Subsampling for ICE Analysis ---
        # We subsample the training data to avoid massive computation and file sizes.
        ice_num_samples = config.get('ICE_NUM_SAMPLES', 50)
        if len(X_train) > ice_num_samples:
            print(f"Subsampling training data for ICE analysis (using {ice_num_samples} samples)...")
            X_ice = X_train.sample(n=ice_num_samples, random_state=seed)
        else:
            X_ice = X_train.copy()
            
        ice_grid_res = config.get('ICE_GRID_RESOLUTION', 50)

        # --- 1D-ICE Calculations ---
        perform_1d_ice_analysis(
            model_wrapper, X_ice, ice_grid_res, report_path, 
            model_name, dataset_name, seed, relative_path
        )

        # --- H-Statistic and 2D-ICE Calculations ---
        # NOTE: H-Statistic and ICE calculations share calculations. Here this is not utilized but could be optimized.
        h_stat_threshold = config.get('PAIRWISE_H_STAT_THRESHOLD', 0.05)
        perform_interaction_analysis(
            model_wrapper, X_train, X_ice, ice_num_samples, ice_grid_res, 
            report_path, model_name, dataset_name, seed, relative_path, h_stat_threshold
        )


        # --- Phase 4: Augmenting Data for New Model Training ---
        print("\n--- Starting Phase 4: Symbolic Regression for Feature Augmentation ---")
        
        # Update config with current seed for PySR determinism
        config["PYSR_RANDOM_STATE"] = seed

        # 1. Identify single feature augmentations
        from src.symbolic_regression import analyze_1d_ice, analyze_2d_ice
        print("Analyzing 1D-ICE data for unary transformations...")
        unary_results = analyze_1d_ice(report_path, model_name, config)
        
        # 2. Identify feature pair augmentations
        print("Analyzing 2D-ICE data for binary interactions...")
        binary_results = analyze_2d_ice(report_path, model_name, config)
        
        # Save discovered equations to a report file
        equations_report_path = os.path.join(report_path, f"discovered_equations_{model_name}.txt")
        with open(equations_report_path, "w") as f:
            f.write("Unary Transformations (1D ICE):\n")
            for feat, eq in unary_results.items():
                f.write(f"{feat}: {eq}\n")
            f.write("\nBinary Interactions (2D ICE):\n")
            for pair, eq in binary_results.items():
                f.write(f"{pair}: {eq}\n")
        
        print(f"Saved discovered equations to {equations_report_path}")

        # 3. Augment Dataset
        print("Augmenting dataset based on discovered equations...")
        save_dir = os.path.join(config['AUGMENTED_DATA_DIR'], relative_path)
        augment_dataset(raw_data_path, unary_results, binary_results, save_dir)
        
    print(f"--- Finished all training runs for {dataset_name} ---")


def main():
    """
    Orchestrates the entire project pipeline.
    Defines which datasets to run experiments on and executes the pipeline for each.
    """
    print("========================================")
    print("         STARTING FULL PIPELINE         ")
    print("========================================") 

    print(f"Using device: {config['DEVICE']}")

    # --- Define which datasets to process ---
    search_path = os.path.join(config['RAW_DATA_DIR'], "**", "*.csv")
    datasets_to_process = glob.glob(search_path, recursive=True)

    if not datasets_to_process:
        print("No raw data files found to process. Exiting.")
        return

    print(f"\nFound {len(datasets_to_process)} datasets to process.")

    # --- Run the iterative training, auto-feature augmentation pipeline for each dataset ---
    for dataset_path in datasets_to_process:
        seeds = MASTER_RANDOM_SEEDS.values()
        run_single_experiment(dataset_path, config, seeds=seeds)

    print("\n========================================")
    print("        FULL PIPELINE COMPLETED         ")
    print("========================================")

def anaylse_pipeline_results():
    # Measure final model performance, generate plots and reports
    # Compare final model to baseline (error in ID and OOD data and d-ICE plots)
    pass

if __name__ == "__main__":
    # If no data is present, generate it
    generate_all_data()
    
    main()

    anaylse_pipeline_results()