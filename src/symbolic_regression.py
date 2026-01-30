# src/symbolic_regression.py

import pandas as pd
import numpy as np
import os
import itertools
import sympy
from collections import Counter

# Handle optional PySR dependency
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    print("Warning: PySR not installed. Symbolic regression phases will be skipped.")

def run_pysr(X, y, config):
    """
    Runs PySRRegressor on the provided data.
    X should be a DataFrame so PySR can infer variable names.
    """
    if not PYSR_AVAILABLE:
        return None
    
    model = PySRRegressor(
        niterations=config.get('PYSR_ITERATIONS', 20),
        binary_operators=config.get('PYSR_BINARY_OPS', ["+", "-", "*", "/"]),
        unary_operators=config.get('PYSR_UNARY_OPS', ["sin", "cos", "exp", "log", "sqrt"]),
        maxsize=config.get('PYSR_MAX_SIZE', 7),
        model_selection="best",
        verbosity=0,
        temp_equation_file=False,
        delete_tempfiles=True,
        random_state=42,
        deterministic=True
    )
    
    # X is expected to be a DataFrame here
    model.fit(X, y)
    return model

def get_skeleton(expr):
    """
    Extracts the functional form of a sympy expression by removing coefficients and additive constants.
    Example: 3.5 * x**2 + 2.1 -> x**2
    Example: 2 * sin(x) -> sin(x)
    """
    # Expand to handle factored forms like 2*(x+1) -> 2x + 2
    expr = sympy.expand(expr)
    
    # Handle 0
    if expr == 0:
        return sympy.sympify("0")

    # Iterate over terms in a sum
    if isinstance(expr, sympy.Add):
        terms = expr.args
    else:
        terms = [expr]
        
    new_terms = []
    for term in terms:
        # Remove numeric coefficients
        # as_coeff_Mul returns (coeff, rest). e.g., 3*x**2 -> (3, x**2)
        coeff, rest = term.as_coeff_Mul()
        
        # If the rest is just number 1 (meaning the term was a constant), ignore it
        if rest == 1:
            continue
            
        new_terms.append(rest)
        
    if not new_terms:
        return sympy.sympify("0")
        
    # Reconstruct the expression sum
    return sum(new_terms)

def analyze_1d_ice(report_path, model_name, config):
    """
    Loads unstandardized 1D ICE data and runs PySR on each feature.
    Returns a dictionary of {feature: equation_string}.
    """
    file_path = os.path.join(report_path, f"unstand_1d-ICE_{model_name}.csv")
    if not os.path.exists(file_path):
        print(f"  - File not found: {file_path}")
        return {}

    print(f"  - Loading 1D ICE data from {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path)
    
    # Identify feature columns (exclude metadata)
    exclude = ['prediction', 'instance', 'Unnamed: 0']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    results = {}
    agreement_threshold = config.get('SYMBOLIC_AGREEMENT_THRESHOLD', 0.9)
    
    for feature in feature_cols:
        # Filter for rows where this feature is active (not NaN)
        sub_df = df[df[feature].notna()]
        if sub_df.empty: continue
        
        instance_ids = sub_df['instance'].unique()
        skeletons = []
        
        print(f"    - Running Symbolic Regression on {len(instance_ids)} ICE curves for feature: {feature}")
        
        for instance_id in instance_ids:
            instance_data = sub_df[sub_df['instance'] == instance_id]
            X = instance_data[[feature]]
            y = instance_data['prediction'].values
            
            model = run_pysr(X, y, config)
            if model:
                eq = model.sympy()
                skel = get_skeleton(eq)
                skeletons.append(skel)
        
        if not skeletons:
            continue
            
        # Analyze consensus
        counts = Counter(skeletons)
        most_common_skel, count = counts.most_common(1)[0]
        frequency = count / len(skeletons)
        
        if frequency >= agreement_threshold:
             print(f"      Consensus found ({frequency:.2%}): {most_common_skel}")
             results[feature] = str(most_common_skel)
        else:
             print(f"      No consensus (Max: {frequency:.2%} for {most_common_skel}). Discarding.")
            
    return results

def analyze_2d_ice(report_path, model_name, config):
    """
    Loads unstandardized 2D ICE data and runs PySR on interacting pairs.
    Returns a dictionary of {pair_name: equation_string}.
    """
    file_path = os.path.join(report_path, f"unstand_2d-ICE_{model_name}.csv")
    if not os.path.exists(file_path):
        print(f"  - File not found: {file_path}")
        return {}

    print(f"  - Loading 2D ICE data from {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path)
    
    exclude = ['prediction', 'instance', 'Unnamed: 0']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    results = {}
    agreement_threshold = config.get('SYMBOLIC_AGREEMENT_THRESHOLD', 0.9)
    
    # Find pairs that exist in the dataframe (joint non-NaNs)
    for f1, f2 in itertools.combinations(feature_cols, 2):
        mask = df[f1].notna() & df[f2].notna()
        if not mask.any():
            continue
            
        sub_df = df[mask]
        pair_name = f"{f1}_&_{f2}"
        
        instance_ids = sub_df['instance'].unique()
        skeletons = []
        
        print(f"    - Running Symbolic Regression on {len(instance_ids)} ICE surfaces for pair: {pair_name}")
        
        for instance_id in instance_ids:
            instance_data = sub_df[sub_df['instance'] == instance_id]
            X = instance_data[[f1, f2]]
            y = instance_data['prediction'].values
            
            model = run_pysr(X, y, config)
            if model:
                eq = model.sympy()
                skel = get_skeleton(eq)
                skeletons.append(skel)
        
        if not skeletons:
            continue
            
        counts = Counter(skeletons)
        most_common_skel, count = counts.most_common(1)[0]
        frequency = count / len(skeletons)
        
        if frequency >= agreement_threshold:
             print(f"      Consensus found ({frequency:.2%}): {most_common_skel}")
             results[pair_name] = str(most_common_skel)
        else:
             print(f"      No consensus (Max: {frequency:.2%} for {most_common_skel}). Discarding.")
            
    return results
