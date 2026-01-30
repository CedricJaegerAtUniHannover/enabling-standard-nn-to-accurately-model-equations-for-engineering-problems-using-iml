# pipelineconfig.py
"""
Configuration for the introduced pipeline.
"""

import torch

# --- Pipeline Configuration ---
config = {
    "RAW_DATA_DIR": "data/01_raw",   # Directory containing raw input data CSV files (will iteratre through subdirectories)
    # TODO: "PROCESSED_DATA_DIR": "data/02_processed",
    # TODO: "AUGMENTED_DATA_DIR": "data/03_augmented",
    "BASELINE_MODELS_DIR": "models/01_baseline",
    "REPORTS_DIR": "outputs/reports",
    "FIGURES_DIR": "outputs/figures",
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # Model Hyperparameters
    "HIDDEN_SIZES": [64, 64, 64],
    "EPOCHS": 200,
    "LEARNING_RATE": 0.001,
    "PATIENCE": 15,
    "BATCH_SIZE": 32,
    # Iteration IML
    "ICE_NUM_SAMPLES": 50,      # Number of instances to sample for ICE plots (reduces N dimension)
    "ICE_GRID_RESOLUTION": 50,  # Number of grid points per axis (reduces M dimension)
    "PAIRWISE_H_STAT_THRESHOLD": 0.05,
    # Symbolic Regression (PySR)
    "PYSR_ITERATIONS": 20,
    "PYSR_BINARY_OPS": ["+", "-", "*", "/", "pow"],
    "PYSR_UNARY_OPS": ["sin", "cos", "exp", "log", "sqrt", "abs"],
    "PYSR_MAX_SIZE": 7, # Limit complexity for quick, rough estimations
    "SYMBOLIC_AGREEMENT_THRESHOLD": 0.90 # Require 90% agreement among ICE curves
}
