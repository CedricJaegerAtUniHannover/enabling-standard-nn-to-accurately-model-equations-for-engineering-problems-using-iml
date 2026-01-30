# src/h_statistic.py
"""
Friedman's H-statistic:
Friedman, J. H., & Popescu, B. E. (2008). Predictive learning via rule ensembles.
The Annals of Applied Statistics, 2(3), 916-954.

Used to measure the strength of interaction between features in a predictive model.

The implementation is based on the `pyartemis` library:
https://github.com/pyartemis/artemis
"""

import pandas as pd
from artemis.interactions_methods.model_agnostic.partial_dependence_based._friedman_h_statistic import FriedmanHStatistic
import matplotlib.pyplot as plt
import seaborn as sns

def get_friedman_h_statistic(model, X, features=None):
    """
    Calculates Friedman's H-statistic for feature interactions.

    This function is a wrapper around the pyartemis library to compute
    Friedman's H-statistic, based on the paper by Friedman and Popescu (2008).

    Parameters
    ----------
    model : object
        A trained machine learning model that has a `predict` method.
    X : pd.DataFrame or np.ndarray
        The data on which to compute the H-statistic.
    features : list of str, optional
        A list of feature names for which to calculate interactions. If None,
        interactions will be calculated for all features. The default is None.

    Returns
    -------
    overall_h_stats : pd.Series
        A series containing the overall H-statistic for each feature.
    pairwise_h_stats : pd.DataFrame
        A DataFrame containing the pairwise H-statistic between features.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    if features is None:
        features = X.columns.tolist()

    h_statistic = FriedmanHStatistic(model, X, feature_names=features)

    overall_h_stats = h_statistic.get_overall_interaction_strength()
    pairwise_h_stats = h_statistic.get_pairwise_interaction_strength()

    return overall_h_stats, pairwise_h_stats


def plot_friedman_h_statistic(pairwise_h_stats, save_path=None):
    """
    Plots the pairwise Friedman's H-statistic as a heatmap.

    Parameters
    ----------
    pairwise_h_stats : pd.DataFrame
        A DataFrame containing the pairwise H-statistic between features.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(pairwise_h_stats, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Pairwise Friedman's H-Statistic Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.show()
    if save_path:
        plt.savefig(save_path)


