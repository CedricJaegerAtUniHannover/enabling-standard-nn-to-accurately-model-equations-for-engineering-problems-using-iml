# src/custom_data_generation/generator.py

import os
import sys
import numpy as np
import pandas as pd

# Add the project root to the Python path to allow for absolute imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.custom_data_generation.formulas import FORMULA_SPECS
from src.config import DATA_GENERATION_SEED

# Define the output directory relative to this script's location
BASE_SYNTH_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', '01_raw', 'synthetic_data')
OUTPUT_NOISE_DIR = os.path.join(BASE_SYNTH_DIR, 'output_noise')
INPUT_NOISE_DIR = os.path.join(BASE_SYNTH_DIR, 'input_noise')

def generate_data(seed=None):
    """
    Generates synthetic datasets based on specifications in `formulas.py` and saves them as CSV files.

    For each specification, this script generates TWO datasets:
    1.  'Output Noise' Version:
        - Inputs (X) are the clean, true values.
        - Output (y) has noise added to the true calculated value.
        - This simulates measurement error on the output.

    2.  'Input Noise' Version:
        - Inputs (X) have noise added to the true values, as specified for each variable.
        - Output (y) is the same noisy output from the 'Output Noise' version.
        - This simulates sensor error on the inputs. The filename will include an `inputnoise-true` tag.

    The ground truth equation is always calculated using the CLEAN inputs.
    """
    if seed is None:
        seed = 42

    # Set the seed for reproducibility across all random operations in this script
    np.random.seed(seed)
    print(f"Using random seed: {seed}")

    # Ensure the output directories exist
    os.makedirs(OUTPUT_NOISE_DIR, exist_ok=True)
    os.makedirs(INPUT_NOISE_DIR, exist_ok=True)
    print(f"Base output directory: {os.path.abspath(BASE_SYNTH_DIR)}")

    for spec in FORMULA_SPECS:
        name = spec['name']
        equation = spec['equation']
        variables = spec['variables']
        num_samples = spec['num_samples']
        output_noise_params = spec['output_noise']

        print(f"\nGenerating datasets for: '{name}'...")

        # --- 1. Generate Clean Input Features (X_true) ---
        feature_columns = []
        for var_name, var_props in variables.items():
            low, high = var_props['range']
            feature_columns.append(np.random.uniform(low, high, num_samples))
        
        X_true = np.stack(feature_columns, axis=1)

        # --- 2. Calculate True Output (y_true) from CLEAN inputs ---
        y_true = equation(X_true)

        # --- 3. Add Noise to the Output to create y_noisy ---
        if output_noise_params['type'] == 'gaussian':
            output_noise = np.random.normal(
                loc=output_noise_params['mean'],
                scale=output_noise_params['std_dev'],
                size=num_samples
            )
        else:
            output_noise = 0
        
        y_noisy = y_true + output_noise

        # --- 4. Generate Noisy Input Features (X_noisy) ---
        noisy_feature_columns = []
        for i, (var_name, var_props) in enumerate(variables.items()):
            # Start with the clean data
            noisy_column = X_true[:, i].copy()
            # Add noise if specified for this variable
            if 'noise' in var_props:
                input_noise_params = var_props['noise']
                if input_noise_params['type'] == 'gaussian':
                    input_noise = np.random.normal(
                        loc=input_noise_params['mean'],
                        scale=input_noise_params['std_dev'],
                        size=num_samples
                    )
                    noisy_column += input_noise
            noisy_feature_columns.append(noisy_column)
        
        X_noisy = np.stack(noisy_feature_columns, axis=1)

        # --- 5. Construct Filenames and Save DataFrames ---
        column_names = list(variables.keys())
        num_vars = len(variables)
        output_noise_tag = f"noise-{output_noise_params['std_dev']:.2f}"
        is_input_noise_present = any('noise' in v for v in variables.values())

        # Define base components for the filename
        base_name_parts = [
            f"SYNTH_{name}",
            f"vars-{num_vars}",
            f"samples-{num_samples}"
        ]
        
        # Version 1: Clean Inputs, Noisy Output
        df_output_noise = pd.DataFrame(X_true, columns=column_names)
        df_output_noise['y'] = y_noisy
        
        filename_output_noise = "_".join(base_name_parts + [output_noise_tag]) + ".csv"
        filepath_output_noise = os.path.join(OUTPUT_NOISE_DIR, filename_output_noise)
        df_output_noise.to_csv(filepath_output_noise, index=False)
        print(f"  - Saved 'output noise' version to: {os.path.join('output_noise', filename_output_noise)}")

        # Version 2: Clean Output, Noisy Inputs
        df_input_noise = pd.DataFrame(X_noisy, columns=column_names)
        df_input_noise['y'] = y_true
        
        input_noise_name_parts = base_name_parts + (["inputnoise-true"] if is_input_noise_present else [output_noise_tag])
        filename_input_noise = "_".join(input_noise_name_parts) + ".csv"
        filepath_input_noise = os.path.join(INPUT_NOISE_DIR, filename_input_noise)
        df_input_noise.to_csv(filepath_input_noise, index=False)
        print(f"  - Saved 'input noise' version to: {os.path.join('input_noise', filename_input_noise)}")
        # Version 2: Noisy Inputs, Noisy Output (only if input noise is specified)
        if is_input_noise_present:
            df_input_noise = pd.DataFrame(X_noisy, columns=column_names)
            df_input_noise['y'] = y_noisy
            
            input_noise_name_parts = base_name_parts + ["inputnoise-true", output_noise_tag]
            filename_input_noise = "_".join(input_noise_name_parts) + ".csv"
            filepath_input_noise = os.path.join(INPUT_NOISE_DIR, filename_input_noise)
            df_input_noise.to_csv(filepath_input_noise, index=False)
            print(f"  - Saved 'input noise' version to: {os.path.join('input_noise', filename_input_noise)}")


if __name__ == "__main__":
    from src.config import DATA_GENERATION_SEED
    seed = DATA_GENERATION_SEED
    generate_data(seed=seed) # Use fallback seed from config