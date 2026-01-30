# src/config.py
"""
Configuration for the project internals, primarily seeds for reproducibility and paths.
"""

# --- SEEDS ---
# Seed for data generation
DATA_GENERATION_SEED = 42

# Seed for data preprocessing (train/val split), training etc.
MASTER_RANDOM_SEEDS = {
    'instance_1': 1
}
"""   
MASTER_RANDOM_SEEDS = {
    'instance_1': 469,
    'instance_2': 108,
    'instance_3': 876,
    'instance_4': 95,
    'instance_5': 745,
    'instance_6': 874,
    'instance_7': 714,
    'instance_8': 825,
    'instance_9': 359,
    'instance_10': 42
}
"""

# --- Fallback Seeds ---
# Fallback seed for data preprocessing (train/val split)
DATA_PREPROCESSING_FALLBACK_SEED = 123

# --- Paths ---
# Path to the raw data directory
RAW_DATA_DIR = "data/01_raw"