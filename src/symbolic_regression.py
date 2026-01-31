# src/symbolic_regression.py

import pandas as pd
import numpy as np
import os
import itertools
import sympy
from collections import Counter
from pysr import PySRRegressor

def run_pysr(X, y, config):
    """
    Runs PySRRegressor on the provided data.
    X should be a DataFrame so PySR can infer variable names.
    """
    
    global_config = config.get('PYSR_GLOBAL', {})
    
    model = PySRRegressor(
        niterations=global_config.get('iterations', 20),
        binary_operators=config.get('binary_ops', ["+", "-", "*", "/"]),
        unary_operators=config.get('unary_ops', []),
        maxsize=config.get('max_size', 7),
        constraints=config.get('constraints', None),
        nested_constraints=config.get('nested_constraints', None),
        model_selection="best",
        verbosity=0,
        temp_equation_file=False,
        delete_tempfiles=True,
        random_state=global_config.get('random_state', 42),
        deterministic=global_config.get('deterministic', True),
        parallelism=global_config.get('paralellism', 'serial')
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

def get_core_structure(expr):
    """
    Recursively removes all additive and multiplicative constants to find the core functional form.
    Used to identify if the underlying law is the same despite different shifts/scales.
    Examples:
    3*x + 5 -> x
    sin(2*x - 1) -> sin(x)
    3 * exp(0.5 * x) -> exp(x)
    """
    if expr.is_Number:
        return expr

    # Handle Add and Mul: remove constant terms/factors
    if isinstance(expr, (sympy.Add, sympy.Mul)):
        args = [get_core_structure(arg) for arg in expr.args if not arg.is_Number]
        if not args:
            return sympy.S.One if isinstance(expr, sympy.Mul) else sympy.S.Zero
        if len(args) == 1:
            return args[0]
        return expr.func(*args)
    
    # Handle Functions (sin, exp, etc) and Pow: Recurse on arguments
    if isinstance(expr, (sympy.Function, sympy.Pow)):
        args = [get_core_structure(arg) for arg in expr.args]
        return expr.func(*args)
        
    return expr

def extract_inner_terms(expr):
    """
    Extracts inner non-linear terms from a unary expression to aid iterative discovery.
    Example: sin(x**2) -> returns [x**2]
    """
    inner_terms = []
    # Traverse the expression tree
    for node in sympy.preorder_traversal(expr):
        # If we find a Power (x^k) or a Function (sin, cos, etc)
        if isinstance(node, (sympy.Pow, sympy.Function)):
            # If it's not the top-level expression itself
            if node != expr:
                # We consider it a building block
                inner_terms.append(node)
    return inner_terms

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
    
    # Load 1D specific configuration
    config_1d = config.get('PYSR_1D_CONFIG', {})
    config_1d['PYSR_GLOBAL'] = config.get('PYSR_GLOBAL', {}) # Pass global settings

    for feature in feature_cols:
        # Filter for rows where this feature is active (not NaN)
        sub_df = df[df[feature].notna()]
        if sub_df.empty: continue
        
        instance_ids = sub_df['instance'].unique()
        skeletons = []
        core_structures = []
        
        print(f"    - Running Symbolic Regression on {len(instance_ids)} ICE curves for feature: {feature}")
        
        for instance_id in instance_ids:
            instance_data = sub_df[sub_df['instance'] == instance_id]
            X = instance_data[[feature]]
            y = instance_data['prediction'].values
            
            # Note: We do NOT center y here anymore. We allow PySR to find the shift (y = f(x) + C).
            # We will strip the shift later using get_skeleton/get_core_structure.
            
            model = run_pysr(X, y, config_1d)
            if model:
                eq = model.sympy()
                skel = get_skeleton(eq)
                core = get_core_structure(skel)
                print(f"      Instance {instance_id}: Eq: {eq} -> Skel: {skel} -> Core: {core}")
                skeletons.append(skel)
                core_structures.append(core)
        
        if not skeletons:
            continue
            
        # 1. Check consensus on Skeleton (Outer shifts removed, inner shifts kept)
        counts = Counter(skeletons)
        most_common_skel, count = counts.most_common(1)[0]
        frequency = count / len(skeletons)
        
        final_eq = None
        
        if frequency >= agreement_threshold:
             print(f"      Exact Consensus found ({frequency:.2%}): {most_common_skel}")
             final_eq = most_common_skel
        else:
             # 2. Check consensus on Core Structure (All shifts/scales removed)
             counts_core = Counter(core_structures)
             most_common_core, count_core = counts_core.most_common(1)[0]
             freq_core = count_core / len(core_structures)
             if freq_core >= agreement_threshold:
                 print(f"      Core Consensus found ({freq_core:.2%}): {most_common_core} (Abandoned shifts/factors)")
                 final_eq = most_common_core
        
        if final_eq:
             results[feature] = str(final_eq)
             # Extract inner terms to help future iterations find complex interactions
             # e.g., if we found sin(x^2), we also want to suggest x^2 as a feature
             inner_terms = extract_inner_terms(final_eq)
             for term in inner_terms:
                 print(f"      -> Extracting inner term for iteration: {term}")
                 results[f"{feature}_inner_{term}"] = str(term)
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
    
    # Load 2D specific configuration
    config_2d = config.get('PYSR_2D_CONFIG', {})
    config_2d['PYSR_GLOBAL'] = config.get('PYSR_GLOBAL', {})
    
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
            
            # Center the target variable
            y = y - np.mean(y)
            
            model = run_pysr(X, y, config_2d)
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
