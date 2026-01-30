# src/data_gen.py

import os
import sys

# Add the project root to the Python path to allow for absolute imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.custom_data_generation.generator import generate_data
from src.config import DATA_GENERATION_SEED

def generate_data():
    """
    Main entry point for generating all project data.
    This script calls the specific data generators with the correct seeds.
    """
    print("--- Starting Synthetic Data Generation ---")
    generate_data(seed=DATA_GENERATION_SEED)
    print("\n--- Data Generation Complete ---")

    # TODO: Implement more data generators if current generator and found datasets are insufficient.

if __name__ == "__main__":
    generate_data()
