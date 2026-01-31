# Model Naming Conventions

This document outlines the standardized naming convention for models in this project, as implemented by the `get_short_model_name` function in ``src/utils.py``. This function generates a short, consistent name for model artifacts.

## Structure

The general structure of a model name is as follows:

-   **short_dataset_name**: A shortened version of the dataset name.
-   **ann**: Stands for Artificial Neural Network.
-   **short_model_details**: Shortened details of the model's architecture.
-   **seed**: The random seed used for training.

---

### Component Breakdown

1.  **`{short_dataset_name}` (from `dataset_name`)**
    *   This part is derived from the original dataset name by applying a series of shortening rules.

    | Original String           | Becomes         |
    | ------------------------- | --------------- |
    | `SYNTH_`                  | `S_`            |
    | `_samples-10000`          | (removed)       |
    | `_inputnoise-true`        | `_in-noise`     |
    | `_noise-`                 | `_out-noise-`   |
    | `_vars-`                  | `-`             |
    | `complex_interaction`     | `complex`       |
    | `trigonometric`           | `trig`          |
    | `polynomial`              | `poly`          |
    | `linear`                  | `lin`           |
    | `rational`                | `rat`           |

2.  **`ann`**
    *   This is a fixed string indicating the model is an Artificial Neural Network.

3.  **`{short_model_details}` (from `model_details`)**
    *   This part is derived from the model's architecture details.

    | Original String | Becomes |
    | --------------- | ------- |
    | `layers-`       | `l-`    |
    | `neurons-`      | `n-`    |

4.  **`s-{seed}` (from `seed`)**
    *   The random seed used for the training run, prefixed with `_s-`.
    *   **Example:** For a seed of `42`, this becomes `_s-42`.

---

### Example

Given the following inputs:
- **`dataset_name`**: `"SYNTH_polynomial_vars-3_samples-10000_inputnoise-true"`
- **`model_details`**: `"layers-3_neurons-64-64-64"`
- **`seed`**: `42`

The `get_short_model_name` function would produce the following model name:

`S_polynomial-3_in-noise_ann_l-3_n-64-64-64_s-42`