# src/ice.py
"""
Individual Conditional Expectation (ICE) plots.

This module provides functions to calculate and plot 1D and 2D ICE plots,
which are used to visualize the behavior of a model's predictions for
individual instances as features are varied.

Individual Conditional Expectation (ICE):
Goldstein, A., Kapelner, A., Bleich, J., & Pitkin, E. (2015).
Peeking inside the black box: Visualizing statistical learning with plots of
individual conditional expectation. Journal of Computational and Graphical
Statistics, 24(1), 44-65.


The implementation is based of the third iML exercise with 2D-ICE beeing the
natural extension of the 1D-ICE concept.

For visualizing the interaction between two features. The 3D plotting is done
using matplotlib's mplot3d toolkit.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_ice_curves(model, X, feature, centered=False):
    """
    Calculates 1D Individual Conditional Expectation (ICE) curves.

    For each instance in X, this function generates a curve by varying the
    value of the specified feature across its unique values in the dataset,
    while keeping other features constant.

    Parameters
    ----------
    model : object
        A trained machine learning model with a `predict` method.
    X : pd.DataFrame
        The data on which to compute the ICE curves.
    feature : str
        The name of the feature to vary.
    centered : bool, optional
        If True, the ICE curves are centered at their first value.
        Default is False.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the ICE curves, with columns for the feature,
        the prediction, and the instance identifier.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

    feature_values = np.sort(X[feature].unique())
    
    ice_data = []

    for index, instance in X.iterrows():
        instance_data = pd.DataFrame([instance.values] * len(feature_values), columns=X.columns)
        instance_data[feature] = feature_values
        
        predictions = model.predict(instance_data)
        
        df = pd.DataFrame({
            feature: feature_values,
            'prediction': predictions
        })
        df['instance'] = index
        ice_data.append(df)
        
    ice_df = pd.concat(ice_data, ignore_index=True)

    if centered:
        ice_df['prediction'] -= ice_df.groupby('instance')['prediction'].transform('first')
        
    return ice_df


def get_ice_surfaces(model, X, features, centered=False):
    """
    Calculates 2D Individual Conditional Expectation (ICE) surfaces.

    For each instance in X, this function generates a surface by varying the
    values of two specified features across a grid of their unique values.

    Parameters
    ----------
    model : object
        A trained machine learning model with a `predict` method.
    X : pd.DataFrame
        The data on which to compute the ICE surfaces.
    features : list of str
        A list containing the names of the two features to vary.
    centered : bool, optional
        If True, the ICE surfaces are centered at their first value.
        Default is False.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the ICE surfaces, with columns for the two
        features, the prediction, and the instance identifier.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
    if len(features) != 2:
        raise ValueError("Exactly two features must be provided for 2D ICE.")

    feature1_vals = np.sort(X[features[0]].unique())
    feature2_vals = np.sort(X[features[1]].unique())
    
    grid = np.array(np.meshgrid(feature1_vals, feature2_vals)).T.reshape(-1, 2)
    
    ice_data = []

    for index, instance in X.iterrows():
        instance_data = pd.DataFrame([instance.values] * len(grid), columns=X.columns)
        instance_data[features[0]] = grid[:, 0]
        instance_data[features[1]] = grid[:, 1]
        
        predictions = model.predict(instance_data)
        
        df = pd.DataFrame(grid, columns=features)
        df['prediction'] = predictions
        df['instance'] = index
        ice_data.append(df)
        
    ice_df = pd.concat(ice_data, ignore_index=True)
    
    if centered:
        ice_df['prediction'] -= ice_df.groupby('instance')['prediction'].transform('first')

    return ice_df


def plot_ice_curves(ice_df, feature, output_name="Prediction", save_path=None):
    """
    Plots 1D ICE curves from the calculated ICE data.

    Parameters
    ----------
    ice_df : pd.DataFrame
        The DataFrame containing ICE curve data from `get_ice_curves`.
    feature : str
        The name of the feature for which the curves are plotted.
    output_name : str, optional
        The label for the y-axis. Default is "Prediction".
    save_path : str, optional
        If provided, the plot will be saved to this path. Default is None.
    """
    plt.figure(figsize=(10, 8))
    
    for instance in ice_df['instance'].unique():
        instance_df = ice_df[ice_df['instance'] == instance]
        plt.plot(instance_df[feature], instance_df['prediction'], 'b-', alpha=0.3)

    plt.title(f"ICE Plot for {feature}")
    plt.xlabel(feature)
    plt.ylabel(output_name)
    
    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_ice_surfaces(ice_df, features, output_name="Prediction", save_path=None):
    """
    Plots individual 2D ICE surfaces in a 3D plot.

    Warning: Plotting a large number of surfaces can be slow and result in a
    cluttered plot.

    Parameters
    ----------
    ice_df : pd.DataFrame
        The DataFrame containing ICE surface data from `get_ice_surfaces`.
    features : list of str
        A list of the two feature names.
    output_name : str, optional
        The label for the z-axis. Default is "Prediction".
    save_path : str, optional
        If provided, the plot will be saved to this path. Default is None.
    """
    if len(features) != 2:
        raise ValueError("Exactly two features must be provided for 2D ICE plot.")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    instances = ice_df['instance'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(instances)))

    for i, instance in enumerate(instances):
        instance_df = ice_df[ice_df['instance'] == instance]
        
        surface_pivot = instance_df.pivot(index=features[1], columns=features[0], values='prediction')
        
        X_grid, Y_grid = np.meshgrid(surface_pivot.columns.astype(float), surface_pivot.index.astype(float))
        Z_grid = surface_pivot.values
        
        ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.1, color=colors[i])

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(output_name)
    ax.set_title(f"ICE Surfaces for {features[0]} and {features[1]}")

    if save_path:
        plt.savefig(save_path)
        
    plt.show()