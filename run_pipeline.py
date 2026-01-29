# run_pipeline.py

from src.data_gen import generate_data as generate_all_data
from src.data_prepro import preprocess_data

def main():
    """
    Orchestrates the entire data pipeline from generation to preprocessing.
    This script ensures that all steps are run in the correct order with
    the right configurations.
    """
    print("========================================")
    print("  STARTING FULL PIPELINE           ")
    print("========================================") 

    # Step 1: Preprocess Raw Data
    preprocess_data()

    print("\n========================================")
    print("  FULL PIPELINE COMPLETED          ")
    print("========================================")

if __name__ == "__main__":
    # If no data is present, generate it
    # generate_all_data()
    
    main()