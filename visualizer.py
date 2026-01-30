# visualizer.py
"""
Visualization utilities for model interpretation and analysis.
"""
import os
import sys
import pandas as pd
import glob
import itertools

# Add src to path to import from custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from h_statistic import plot_friedman_h_statistic
from ice import plot_ice_curves, plot_ice_surfaces

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'reports')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'figures')

def get_all_report_files(root_dir):
    """Recursively finds all CSV files in the reports directory."""
    return glob.glob(os.path.join(root_dir, '**', '*.csv'), recursive=True)

def main():
    """
    Main function to run the visualization script.
    """
    # Ensure directories exist
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    report_files = get_all_report_files(REPORTS_DIR)

    if not report_files:
        print("No report files found in 'outputs/reports'.")
        print("Please generate some reports first by running the pipeline.")
        return

    print("Please select a file to visualize:")
    # Display relative paths for cleaner output
    rel_paths = [os.path.relpath(f, REPORTS_DIR) for f in report_files]
    
    for i, path in enumerate(rel_paths):
        print(f"{i + 1}: {path}")

    try:
        choice = int(input(f"Enter a number (1-{len(report_files)}): ")) - 1
        if not 0 <= choice < len(report_files):
            raise ValueError
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    selected_file = report_files[choice]
    filename = os.path.basename(selected_file)

    print(f"Loading: {filename}")
    # Assuming the data is stored in csv format.
    try:
        # H-statistic files have an index column, ICE files do not
        if 'h-statistic' in filename:
            df = pd.read_csv(selected_file, index_col=0)
        else:
            df = pd.read_csv(selected_file)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Logic to select visualization based on dataset name
    if 'h-statistic' in filename:
        print("Generating Friedman's H-statistic heatmap...")
        save_path = os.path.join(FIGURES_DIR, f"{os.path.splitext(filename)[0]}_heatmap.png")
        plot_friedman_h_statistic(df, save_path)
        print(f"Plot saved to {save_path}")
        
    elif '1d-ICE' in filename:
        print("Detected 1D-ICE data.")
        centered_input = input("Do you want a centered ICE plot? (y/n): ").lower()
        centered = centered_input == 'y'
        
        # Identify feature columns (exclude 'prediction', 'instance', and any index cols)
        exclude_cols = ['prediction', 'instance', 'Unnamed: 0']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # The dataframe might contain multiple features concatenated (with NaNs for others)
        valid_features = []
        for col in feature_cols:
            if df[col].notna().any():
                valid_features.append(col)
        
        if not valid_features:
            print("No valid feature columns found.")
            return

        print(f"Found features: {valid_features}")
        for feature in valid_features:
            print(f"Plotting ICE for feature: {feature}")
            # Filter data for this feature
            sub_df = df[df[feature].notna()].copy()
            
            save_name = f"{os.path.splitext(filename)[0]}_{feature}"
            if centered:
                save_name += "_centered"
            save_path = os.path.join(FIGURES_DIR, f"{save_name}.png")
            
            plot_ice_curves(sub_df, feature, save_path=save_path, centered=centered)
            print(f"Saved plot to {save_path}")

    elif '2d-ICE' in filename:
        print("Detected 2D-ICE data.")
        centered_input = input("Do you want a centered ICE plot? (y/n): ").lower()
        centered = centered_input == 'y'

        single_instance_input = input("Do you want to plot a single instance? (y/n): ").lower()
        selected_instance = None
        
        if single_instance_input == 'y':
            if 'instance' in df.columns:
                unique_instances = sorted(df['instance'].unique())
                print(f"Available instances (first 5): {unique_instances[:5]}...")
                try:
                    inst_input = input(f"Enter instance ID (default {unique_instances[0]}): ")
                    if inst_input.strip():
                        selected_instance = int(inst_input)
                        if selected_instance not in unique_instances:
                            print(f"Instance {selected_instance} not found. Using {unique_instances[0]}.")
                            selected_instance = unique_instances[0]
                    else:
                        selected_instance = unique_instances[0]
                except ValueError:
                    print(f"Invalid input. Using {unique_instances[0]}.")
                    selected_instance = unique_instances[0]
        
        exclude_cols = ['prediction', 'instance', 'Unnamed: 0']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Find pairs that are not null together
        valid_pairs = []
        for f1, f2 in itertools.combinations(feature_cols, 2):
            # Check if there are rows where both are not NaN
            mask = df[f1].notna() & df[f2].notna()
            if mask.any():
                valid_pairs.append((f1, f2))
        
        if not valid_pairs:
            print("No valid feature pairs found.")
            return

        print(f"Found pairs: {valid_pairs}")
        for f1, f2 in valid_pairs:
            print(f"Plotting ICE surface for: {f1} & {f2}")
            mask = df[f1].notna() & df[f2].notna()
            sub_df = df[mask].copy()
            
            save_name = f"{os.path.splitext(filename)[0]}_{f1}_vs_{f2}"
            if centered:
                save_name += "_centered"
            save_path = os.path.join(FIGURES_DIR, f"{save_name}.png")
            
            plot_ice_surfaces(sub_df, [f1, f2], save_path=save_path, centered=centered, instance_id=selected_instance)
            print(f"Saved plot to {save_path}")

    else:
        print(f"No visualization method defined for '{filename}'.")


if __name__ == "__main__":
    main()