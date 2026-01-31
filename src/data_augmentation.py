# src/data_augmentation.py

import os
import pandas as pd
import sympy
import numpy as np

def augment_dataset(raw_data_path, unary_equations, binary_equations, save_dir):
    """
    Augments the raw dataset with new features derived from symbolic regression equations.

    Args:
        raw_data_path (str): Path to the original raw CSV file.
        unary_equations (dict): Dictionary {feature_name: equation_str} for 1D transformations.
        binary_equations (dict): Dictionary {pair_name: equation_str} for 2D interactions.
        save_dir (str): Directory to save the augmented dataset.

    Returns:
        str: Path to the saved augmented dataset, or None if no augmentation occurred.
    """
    print(f"  - Loading raw data from {raw_data_path}...")
    try:
        df = pd.read_csv(raw_data_path)
    except Exception as e:
        print(f"Error loading raw data: {e}")
        return None

    new_features = {}

    # Helper to safely evaluate equations
    def evaluate_eq(eq_str, context_df):
        try:
            # Parse the equation string into a SymPy expression
            expr = sympy.sympify(eq_str)
            
            # Identify symbols (variables) in the expression
            symbols = sorted(list(expr.free_symbols), key=lambda s: str(s))
            
            if not symbols:
                print(f"    Warning: Equation '{eq_str}' is constant. Skipping.")
                return None

            symbol_names = [str(s) for s in symbols]
            
            # Check if all symbols exist in the DataFrame
            missing_cols = [s for s in symbol_names if s not in context_df.columns]
            if missing_cols:
                print(f"    Warning: Missing columns {missing_cols} for equation '{eq_str}'. Skipping.")
                return None
            
            # Create a lambda function for fast evaluation using NumPy
            f = sympy.lambdify(symbols, expr, modules=['numpy'])
            
            # Prepare arguments (columns) for the function
            args = [context_df[col].values for col in symbol_names]
            
            # Calculate result
            result = f(*args)
            
            # Check for NaNs or Infs (e.g., log of negative numbers)
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                mask_valid = np.isfinite(result)
                if not np.any(mask_valid):
                    print(f"    Warning: Equation '{eq_str}' resulted in all NaNs/Infs. Skipping.")
                    return None
                
                # Impute NaNs/Infs with max/min of valid data
                max_valid = np.max(result[mask_valid])
                min_valid = np.min(result[mask_valid])
                
                result[np.isposinf(result)] = max_valid
                result[np.isneginf(result)] = min_valid
                result[np.isnan(result)] = max_valid
                
                print(f"    Warning: Equation '{eq_str}' contained NaNs/Infs. Imputed with min/max finite values.")
                
            return result
        except Exception as e:
            print(f"    Error evaluating equation '{eq_str}': {e}")
            return None

    # 1. Apply Unary Transformations
    if unary_equations:
        print(f"  - Applying {len(unary_equations)} unary transformations...")
        for feature, eq_str in unary_equations.items():
            new_col_name = f"{feature}_aug"
            print(f"    - Creating {new_col_name} from {eq_str}")
            result = evaluate_eq(eq_str, df)
            if result is not None:
                new_features[new_col_name] = result

    # 2. Apply Binary Interactions
    if binary_equations:
        print(f"  - Applying {len(binary_equations)} binary interactions...")
        for pair_name, eq_str in binary_equations.items():
            new_col_name = f"{pair_name}_aug"
            print(f"    - Creating {new_col_name} from {eq_str}")
            result = evaluate_eq(eq_str, df)
            if result is not None:
                new_features[new_col_name] = result

    # 3. Concatenate and Save
    if new_features:
        new_df = pd.DataFrame(new_features)
        augmented_df = pd.concat([df, new_df], axis=1)
        
        # Ensure 'y' is the last column if it exists
        if 'y' in augmented_df.columns:
            cols = [c for c in augmented_df.columns if c != 'y'] + ['y']
            augmented_df = augmented_df[cols]
        
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.basename(raw_data_path)
        save_path = os.path.join(save_dir, filename)
        
        augmented_df.to_csv(save_path, index=False)
        print(f"  - Saved augmented dataset with {len(new_features)} new features to {save_path}")
        return save_path
    else:
        print("  - No new features generated. Skipping save.")
        return None