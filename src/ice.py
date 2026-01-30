# src/ice.py
"""
1d-ICE according to iML exercise 3 and 2d-ICE according to ... and based on: ...
"""

import sys
sys.path.insert(0, "")

import numpy as np
from utils.dataset import Dataset
from utils.styled_plot import plt


def calculate_ice(model, X, s):
    """
    Iterates over the observations in X and for each observation i, takes the x_s value of i and replaces the x_s
    values of all other observation with this value. The model is used to make a prediction for this new data.
    The data and prediction of each iteration are added to numpy arrays x_ice and y_ice.
    For the current iteration i and the selected feature index s, the following equation is ensured:
    X_ice[i, :, s] == X[i, s]

    Parameters:
        model: Classifier which can call a predict method.
        X (np.ndarray with shape (num_instances, num_features)): Input data.
        s (int): Index of the feature x_s.

    Returns:
        X_ice (np.ndarray with shape (num_instances, num_instances, num_features)): Changed input data w.r.t. x_s.
        y_ice (np.ndarray with shape (num_instances, num_instances)): Predicted data.
    """
    # NOTE: Although to me it would make more sense to have i be the same as the number of ice curves,
    # the test and the X_ice[i, :, s] == X[i, s] condition demand to treat i as the grid points.
    # That way when I look at a grid point i, for every ice curve the value for the certain feature s is the
    # the same, according to the condition, the value that that feature has in the instance - so for
    # every instance we create a grid point i.

    # Extract unique values for the grid
    #grid_points = np.unique(X[:, s])
    # The test does not demand uniqueness, therefore I replace this with the following line:
    grid_points = X[:, s]
    #num_grid_points = len(grid_points) # replaced for simpler oneliner below
    num_grid_points, num_features = X.shape
    # print(f"Number of grid points for feature {s}: {num_grid_points}, number of features: {num_features}")
    
    # Identify amount of ice curves (i.e., number of instances)
    num_ice_curves = X.shape[0]
    # print(f"Number of ice curves i.e. number of instances: {num_ice_curves}")

    X_ice = np.zeros((num_grid_points, num_ice_curves, num_features))
    y_ice = np.zeros((num_grid_points, num_ice_curves))

    
    for i in range(num_grid_points):
        for c in range(num_ice_curves):
            X_ice[i, c, :] = X[c, :].copy()
            X_ice[i, c, s] = grid_points[i]
        
        # calculate outputs for the grid points
        y_ice[i, :] = model.predict(X_ice[i, :, :])

    return X_ice, y_ice


def prepare_ice(model, X, s, centered=False):
    """
    Uses `calculate_ice` and iterates over the rows of the returned arrays to obtain as many curves as
    observations.

    Parameters:
        model: Classifier which can call a predict method.
        X (np.ndarray with shape (num_instances, num_features)): Input data.
        s (int): Index of the feature x_s.
        centered (bool): Whether c-ICE should be used.

    Returns:
        all_x (list or 1D np.ndarray): List of lists of the x values.
        all_y (list or 1D np.ndarray): List of lists of the y values.
            Each entry in `all_x` and `all_y` represents one line in the plot.
    """
    X_ice, y_ice = calculate_ice(model, X, s)
    num_grid_points, num_ice_curves = y_ice.shape

    all_x = np.zeros((num_ice_curves, num_grid_points))
    all_y = np.zeros((num_ice_curves, num_grid_points))

    # Sorting X_ice and y_ice according to X_ice grid points, smallest to largest
    sorted_X_ice = np.zeros_like(X_ice[:, :, s])
    sorted_y_ice = np.zeros_like(y_ice)
    for c in range(num_ice_curves):  # for each ice curve (second dimension - see function calculate_ice)
        x_vals = X_ice[:, c, s]
        y_vals = y_ice[:, c]
        
        sort_indices = np.argsort(x_vals) # first dimension is for every grid point
        sorted_X_ice[:, c] = x_vals[sort_indices]
        sorted_y_ice[:, c] = y_vals[sort_indices]

    if centered:
        y_offsets = sorted_y_ice[0, :]  # all the sorted y values at the first grid point
        sorted_y_ice = sorted_y_ice - y_offsets[np.newaxis, :]  # center each ice curve
    
    # NOTE: opposing the function docstrin, I do not use all_x and all_y (1d arrays) for the output,
    # because the test demands 2d arrays
    for c in range(num_ice_curves):  # for each ice curve (second dimension - see function calculate_ice)
        # bring in right shape according to test
        all_x[c, :] = sorted_X_ice[:, c]
        all_y[c, :] = sorted_y_ice[:, c]
        # now we have that the first dimension is for each ice curve and the second for each grid point of it
    '''
    # reshape sorted arrays in 1D arrays for plotting
    for i in range(X_ice.shape[0]): # for each grid point (first dimension - see function calculate_ice)
        all_x = np.append(all_x, sorted_X_ice[i, :, s]) # effectively for as many ice curves there are
        all_y = np.append(all_y, sorted_y_ice[i, :])
    '''
    return all_x, all_y


def plot_ice(model, dataset, X, s, centered=False):
    """
    Creates a plot object and fills it with the content of `prepare_ice`.
    Note: `show` method is not called.

    Parameters:
        model: Classifier which can call a predict method.
        dataset (utils.Dataset): Used dataset to train the model. Required to receive the input and output label.
        s (int): Index of the feature x_s.
        centered (bool): Whether c-ICE should be used.

    Returns: 
        plt (matplotlib.pyplot or utils.styled_plot.plt)
    """
    all_x, all_y = prepare_ice(model, X, s, centered) # 2d arrays each

    feature_name = dataset.get_input_labels()[s]

    plt.figure()

    for c in range(all_x.shape[0]): # for each ice curve
        plt.plot(all_x[c, :], all_y[c,: ], alpha=0.2)
 
    plt.title(f"ICE plot for feature '{feature_name}'")
    plt.xlabel(feature_name)
    plt.ylabel(dataset.get_output_label())

    return plt