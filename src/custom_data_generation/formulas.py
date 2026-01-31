# d:\Users\cedri\OneDrive\enabling-standard-nn-to-accurately-model-equations-for-engineering-problems-using-iml\src\custom_data_generation\formulas.py
import numpy as np

"""
Define the specifications for synthetic dataset generation.

Each entry in FORMULA_SPECS is a dictionary that defines one dataset.
Keys:
    - name (str): A unique, descriptive name for the equation. Used in the filename.
    - equation (lambda): A function that takes a NumPy array `x` of the TRUE (clean) inputs
                         and returns the output `y`. `x` is an (n_samples, n_features) array.
                         Access columns like x[:, 0] for x1, x[:, 1] for x2, etc.
    - variables (dict): Defines the input variables (features).
        - Keys are variable names (e.g., 'x1', 'x2').
        - Values are dictionaries with:
            - 'range' (list): The [min, max] for random generation of the true value.
            - 'noise' (dict, optional): Defines noise to be added to this input variable.
                - 'type' (str): The type of noise (e.g., 'gaussian').
                - 'mean' (float): The mean of the noise distribution.
                - 'std_dev' (float): The standard deviation of the noise.
    - num_samples (int): The number of data points to generate.
    - output_noise (dict): Defines the properties of the noise to be added to the output.
        - 'type' (str): The type of noise (e.g., 'gaussian').
        - 'mean' (float): The mean of the noise distribution.
        - 'std_dev' (float): The standard deviation of the noise. This controls the noise level.
"""

FORMULA_SPECS = [
    {
        'name': 'linear_2d',
        'equation': lambda x: 3.5 * x[:, 0] + 2.0, # y = 3.5*x1 + 2.0
        'variables': {
            'x1': {
                'range': [-10, 10],
                'noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 0.2}
            }
        },
        'num_samples': 10_000,
        'output_noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 1.0}
    },
    {
        'name': 'polynomial_3d',
        'equation': lambda x: 0.5 * x[:, 0]**3 - 2 * x[:, 1]**2 + 3 * x[:, 2], # y = 0.5*x1^3 - 2*x2^2 + 3*x3
        'variables': {
            'x1': {
                'range': [-4, 4],
                'noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 0.1}
            },
            'x2': {
                'range': [-4, 4],
                'noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 0.1}
            },
            'x3': {
                'range': [-4, 4],
                'noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 0.1}
            }
        },
        'num_samples': 10_000,
        'output_noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 1.0}
    },
    {
        'name': 'trigonometric_2d',
        'equation': lambda x: np.sin(x[:, 0]) + np.cos(x[:, 1]), # y = sin(x1) + cos(x2)
        'variables': {
            'x1': {
                'range': [-np.pi, np.pi],
                'noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 0.05}
            },
            'x2': {
                'range': [-np.pi, np.pi]
                # No noise specified for x2, so it will remain clean in the input-noise dataset
            }
        },
        'num_samples': 10_000,
        'output_noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 1.0}
    },
    {
        'name': 'complex_interaction_2d',
        'equation': lambda x: np.exp(-0.5 * x[:, 0]) * (x[:, 1]**2), # y = exp(-0.5*x1) * (x2^2)
        'variables': {
            'x1': {
                'range': [0, 5],
                'noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 0.1}
            },
            'x2': {
                'range': [-3, 3],
                'noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 0.1}
            }
        },
        'num_samples': 10_000,
        'output_noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 1.0}
    },
    {
        'name': 'rational_3d',
        'equation': lambda x: (x[:, 0] * x[:, 1]) / (2 * x[:, 2]), # y = (x1 * x2) / (2 * x3)
        'variables': {
            'x1': {
                'range': [-10, 10],
                'noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 0.1}
            },
            'x2': {
                'range': [-10, 10],
                'noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 0.1}
            },
            'x3': {
                'range': [0.5, 10], # Avoid division by zero
                'noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 0.05}
            }
        },
        'num_samples': 10_000,
        'output_noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 1.0}
    },
    {
        'name': 'Ainteracting_trig',
        'equation': lambda x: np.sin(x[:, 0]**2 * x[:, 1]) + x[:, 1], # y = sin(x1^2 * x2) + x3
        'variables': {
            'x1': {
                'range': [-np.pi, np.pi],
                'noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 0.05}
            },
            'x2': {
                'range': [-np.pi, np.pi],
                'noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 0.05}
            },
            'x3': {
                'range': [-np.pi, np.pi],
                'noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 0.4}
            }
        },
        'num_samples': 10_000,
        'output_noise': {'type': 'gaussian', 'mean': 0.0, 'std_dev': 1.0}
    },
]
