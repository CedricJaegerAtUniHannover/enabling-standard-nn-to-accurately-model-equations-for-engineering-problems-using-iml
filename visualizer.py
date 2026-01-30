# visualizer.py
"""
Visualization utilities for model interpretation and analysis.
"""
import os
import sys
import pandas as pd

# Add src to path to import from custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from h_statistic import plot_friedman_h_statistic

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'reports')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'figures')


def main():
    """
    Main function to run the visualization script.
    """
    # Ensure directories exist
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    available_datasets = [f for f in os.listdir(REPORTS_DIR) if os.path.isfile(os.path.join(REPORTS_DIR, f))]

    if not available_datasets:
        print("No datasets found in 'outputs/reports'.")
        print("Please generate some reports first by running the pipeline.")
        return

    print("Please select a dataset to visualize:")
    for i, dataset_name in enumerate(available_datasets):
        print(f"{i + 1}: {dataset_name}")

    try:
        choice = int(input(f"Enter a number (1-{len(available_datasets)}): ")) - 1
        if not 0 <= choice < len(available_datasets):
            raise ValueError
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    selected_dataset = available_datasets[choice]
    dataset_path = os.path.join(REPORTS_DIR, selected_dataset)

    print(f"Loading dataset: {selected_dataset}")
    # Assuming the data is stored in csv format.
    try:
        df = pd.read_csv(dataset_path, index_col=0)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Logic to select visualization based on dataset name
    if 'h_statistic' in selected_dataset:
        print("Generating Friedman's H-statistic heatmap...")
        save_path = os.path.join(FIGURES_DIR, f"{os.path.splitext(selected_dataset)[0]}_heatmap.png")
        plot_friedman_h_statistic(df, save_path)
        print(f"Plot saved to {save_path}")
    else:
        print(f"No visualization method defined for '{selected_dataset}'.")


if __name__ == "__main__":
    main()