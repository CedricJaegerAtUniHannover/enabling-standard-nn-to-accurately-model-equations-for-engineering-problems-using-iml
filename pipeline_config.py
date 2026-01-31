# pipelineconfig.py
"""
Configuration for the introduced pipeline.
"""

import torch

# --- Pipeline Configuration ---
config = {
    "RAW_DATA_DIR": "data/01_raw",   # Directory containing raw input data CSV files (will iteratre through subdirectories)
    # TODO: "PROCESSED_DATA_DIR": "data/02_processed",
    "AUGMENTED_DATA_DIR": "data/03_augmented_raw",
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
    "SYMBOLIC_AGREEMENT_THRESHOLD": 0.90, # Require 90% agreement among regressions of ICE curves
    "PYSR_GLOBAL": {
        "paralellism": "serial",
        "deterministic": True,
        "random_state": 42,
        "iterations": 10,
    },

    # --- 1D ICE Configuration ---
    # Goal: Find power, saturation, and oscillation laws.
    #
    # Interpretation of Constraints:
    # We want to enable the form: y = A * f(B * x + C) + D
    # where f is a single non-linear operation (unary op or power).
    # 1. "binary_ops" (['+', '-', '*', '/']) allow the affine transformations (A, B, C, D).
    # 2. "nested_constraints" prevent stacking non-linearities (e.g., sin(exp(x)) is forbidden).
    #    This ensures that 'f' operates on a linear transformation of x, and the result is linearly transformed.
    #    Examples allowed: (2*x + 2)^2, sin(5*x - 2), 3*exp(x) + 5
    "PYSR_1D_CONFIG": {
        "binary_ops": ["+", "-", "*", "/", "pow"],
        "unary_ops": ["sin", "cos", "exp", "log", "sqrt"], # Focused set: Power, Saturation/Log, Periodic
        "max_size": 10, # Reduced for speed. Sufficient for A*f(B*x+C)+D (size ~10)
        "constraints": {'pow': (-1, 1)}, # Exponent max complexity 1 (constant)
        "nested_constraints": {
            "sin": {"sin": 0, "cos": 0, "exp": 0, "log": 0, "sqrt": 0},
            "cos": {"sin": 0, "cos": 0, "exp": 0, "log": 0, "sqrt": 0},
            "exp": {"sin": 0, "cos": 0, "exp": 0, "log": 0, "sqrt": 0},
            "log": {"sin": 0, "cos": 0, "exp": 0, "log": 0, "sqrt": 0},
            "sqrt": {"sin": 0, "cos": 0, "exp": 0, "log": 0, "sqrt": 0},
        }
    },

    # --- 2D ICE Configuration ---
    # Goal: Find specific interactions: Product (x*y), Ratio (x/y), Coupled Power (x^y), Compound Phase (sin(x+y)), Frequency Modulation (sin(x*y))
    "PYSR_2D_CONFIG": {
        "binary_ops": ["+", "*", "/", "pow"], # '+' needed for sin(x+y), '*' for sin(x*y)
        "unary_ops": ["sin"], # Only sin needed for phase/frequency interaction
        "max_size": 9,
        "constraints": None,
        "nested_constraints": {
            "sin": {"sin": 0, "pow": 0}, # Prevent deep nesting
            "pow": {"sin": 0, "pow": 0}
        }
    }
}
