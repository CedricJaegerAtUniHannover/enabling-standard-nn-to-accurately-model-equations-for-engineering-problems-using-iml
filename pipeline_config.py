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
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # Model Hyperparameters
    "HIDDEN_SIZES": [64, 64, 64],
    "EPOCHS": 200,
    "LEARNING_RATE": 0.001,
    "PATIENCE": 15,
    "BATCH_SIZE": 32
}
