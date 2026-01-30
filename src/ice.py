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

def get_ice_curves(model, X, feature, centered=False, num_grid_points=100):
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
    num_grid_points : int, optional
        Number of points to evaluate along the feature range. Default is 100.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the ICE curves, with columns for the feature,
        the prediction, and the instance identifier.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

    feature_values = np.linspace(X[feature].min(), X[feature].max(), num_grid_points)
    
    num_instances = len(X)
    num_feature_values = len(feature_values)

    # Repeat each instance for each feature value
    instance_data = X.loc[X.index.repeat(num_feature_values)].reset_index(drop=True)
    
    # Create a tiled array of feature values
    tiled_feature_values = np.tile(feature_values, num_instances)
    
    # Assign the tiled feature values to the feature column
    instance_data[feature] = tiled_feature_values
    
    # Make a single prediction call
    predictions = model.predict(instance_data)
    
    # Create the instance identifiers
    instance_ids = np.repeat(X.index, num_feature_values)

    # Construct the final DataFrame
    ice_df = pd.DataFrame({
        feature: tiled_feature_values,
        'prediction': predictions,
        'instance': instance_ids
    })

    if centered:
        ice_df['prediction'] -= ice_df.groupby('instance')['prediction'].transform('first')
        
    return ice_df


def get_ice_surfaces(model, X, features, centered=False, num_grid_points=50):
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
    num_grid_points : int, optional
        Number of grid points per feature axis. Default is 50 (resulting in 50x50 grid).

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

    feature1_vals = np.linspace(X[features[0]].min(), X[features[0]].max(), num_grid_points)
    feature2_vals = np.linspace(X[features[1]].min(), X[features[1]].max(), num_grid_points)
    
    grid = np.array(np.meshgrid(feature1_vals, feature2_vals)).T.reshape(-1, 2)
    
    num_instances = len(X)
    num_grid_points = len(grid)

    # Repeat each instance for each grid point
    instance_data = X.loc[X.index.repeat(num_grid_points)].reset_index(drop=True)
    
    # Create tiled grid values
    tiled_grid = np.tile(grid, (num_instances, 1))
    
    # Assign tiled grid values to feature columns
    instance_data[features[0]] = tiled_grid[:, 0]
    instance_data[features[1]] = tiled_grid[:, 1]

    # Make a single prediction call
    predictions = model.predict(instance_data)
    
    # Create instance identifiers
    instance_ids = np.repeat(X.index, num_grid_points)

    # Construct the final DataFrame
    ice_df = pd.DataFrame(tiled_grid, columns=features)
    ice_df['prediction'] = predictions
    ice_df['instance'] = instance_ids

    if centered:
        ice_df['prediction'] -= ice_df.groupby('instance')['prediction'].transform('first')

    return ice_df


def plot_ice_curves(ice_df, feature, output_name="Prediction", save_path=None, centered=False):
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
    centered : bool, optional
        If True, plots centered ICE curves (c-ICE), anchored at the feature value closest to 0.
        Default is False.
    """
    plt.figure(figsize=(10, 8))
    
    plot_df = ice_df.copy()
    
    if centered:
        # Center the predictions per instance at the feature value closest to 0
        plot_df['abs_feature'] = plot_df[feature].abs()
        # Find index of row with feature value closest to 0 for each instance
        idx_anchors = plot_df.groupby('instance')['abs_feature'].idxmin()
        # Get the prediction values at these indices
        anchor_predictions = plot_df.loc[idx_anchors, ['instance', 'prediction']].set_index('instance')['prediction']
        # Subtract anchor prediction
        plot_df['prediction'] = plot_df['prediction'] - plot_df['instance'].map(anchor_predictions)
        plot_df = plot_df.drop(columns=['abs_feature'])
        
        title_suffix = " (Centered at 0)"
    else:
        title_suffix = ""

    # Sort by feature for correct line plotting
    plot_df = plot_df.sort_values(by=[feature])

    for instance in plot_df['instance'].unique():
        instance_df = plot_df[plot_df['instance'] == instance]
        plt.plot(instance_df[feature], instance_df['prediction'], 'b-', alpha=0.3)

    plt.title(f"ICE Plot for {feature}{title_suffix}")
    plt.xlabel(feature)
    plt.ylabel(output_name)
    
    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_ice_surfaces(ice_df, features, output_name="Prediction", save_path=None, centered=False, instance_id=None):
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
    centered : bool, optional
        If True, plots centered ICE surfaces (c-ICE), anchored at the feature values closest to (0,0).
        Default is False.
    instance_id : int, optional
        If provided, only plots the surface for this specific instance ID.
    """
    if len(features) != 2:
        raise ValueError("Exactly two features must be provided for 2D ICE plot.")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    plot_df = ice_df.copy()

    if instance_id is not None:
        plot_df = plot_df[plot_df['instance'] == instance_id]

    if centered:
        # Center the predictions per instance at the feature values closest to (0,0)
        plot_df['dist_sq'] = plot_df[features[0]]**2 + plot_df[features[1]]**2
        idx_anchors = plot_df.groupby('instance')['dist_sq'].idxmin()
        anchor_predictions = plot_df.loc[idx_anchors, ['instance', 'prediction']].set_index('instance')['prediction']
        plot_df['prediction'] = plot_df['prediction'] - plot_df['instance'].map(anchor_predictions)
        plot_df = plot_df.drop(columns=['dist_sq'])
        
        title_suffix = " (Centered at 0,0)"
    else:
        title_suffix = ""

    instances = plot_df['instance'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(instances)))

    for i, instance in enumerate(instances):
        instance_df = plot_df[plot_df['instance'] == instance]
        
        surface_pivot = instance_df.pivot(index=features[1], columns=features[0], values='prediction')
        
        X_grid, Y_grid = np.meshgrid(surface_pivot.columns.astype(float), surface_pivot.index.astype(float))
        Z_grid = surface_pivot.values
        
        ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3, color=colors[i])

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(output_name)
    ax.set_title(f"ICE Surfaces for {features[0]} and {features[1]}{title_suffix}")

    # Set a default 3D angular view
    ax.view_init(elev=30, azim=-60)

    if save_path:
        plt.savefig(save_path)
        
    plt.show()