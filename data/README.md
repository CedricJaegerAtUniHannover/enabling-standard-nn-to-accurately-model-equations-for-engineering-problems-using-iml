# Data Naming Convention

This document outlines the standardized naming convention for all datasets used in this project. A consistent naming scheme is crucial for clarity, reproducibility, and automated processing.

## General Structure

All data files should follow this general structure:

`<Source>_<Name>_<Details>.<extension>`

---

### Component Breakdown

1.  **`<Source>` (Required)**
    *   An uppercase identifier for the origin of the data.
    *   **Examples:**
        *   `SYNTH`: For synthetically generated data from custom scripts.
        *   `SRBENCH`: For datasets from the SRBench collection.
        *   `SRSD`: For datasets from the Symbolic Regression for Scientific Discovery benchmark.
        *   `SIMBA`: For data generated from ODEs using `simba-ml`.

2.  **`<Name>` (Required)**
    *   A descriptive, lowercase name for the specific dataset or equation.
    *   **Examples:**
        *   `linear_2d` (for a synthetic 2D linear equation)
        *   `feynman_I_10_7` (for a specific Feynman problem from SRBench)
        *   `srsd_hard_1` (for a specific problem from the SRSD hard tier)

3.  **`<Details>` (Optional but Recommended)**
    *   A series of key-value pairs, separated by underscores, providing metadata about the dataset. The key and value are separated by a hyphen.
    *   This makes file properties easily parsable.
    *   **Common Keys:**
        *   `vars`: Number of variables/features (e.g., `vars-3`).
        *   `samples`: Number of data points (e.g., `samples-1000`).
        *   `noise`: Noise level on the **output** variable, typically standard deviation (e.g., `noise-0.50`).
        *   `inputnoise`: A flag indicating if noise was added to the input variables (e.g., `inputnoise-true`).
        *   `difficulty`: A category like `easy`, `medium`, `hard`.
        *   `irrelevant`: Number of added irrelevant features (e.g., `irrelevant-5`).

4.  **`<extension>` (Required)**
    *   The file format.
    *   **Examples:** `csv`, `parquet`.

---

### Examples

*   **Synthetic Data (Custom Scripts Generated):**
    *   `SYNTH_polynomial_3d_vars-3_samples-10000_noise-1.00.csv` (for clean inputs)
    *   `SYNTH_polynomial_3d_vars-3_samples-10000_inputnoise-true_noise-1.00.csv` (for noisy inputs)

*   **SRBench Data (Hypothetical):**
    *   `SRBENCH_feynman_I_12_4_vars-4.csv`

*   **SRSD Data (Hypothetical):**
    *   `SRSD_feynman_hard_3_irrelevant-5_noise-0.10.csv`
