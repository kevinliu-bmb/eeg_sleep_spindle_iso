import numpy as np
from sklearn.utils import resample


def bootstrap_ci(data, ci=95, n_bootstrap=1000):
    """
    Compute the mean and confidence interval (CI) of input data using bootstrapping.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Input data array where rows represent samples and columns represent features (e.g., frequency bins).

    ci : float, optional (default=95)
        Desired confidence interval percentage. Commonly used values are 90, 95, and 99.

    n_bootstrap : int, optional (default=1000)
        Number of bootstrap resamples to perform.

    Returns
    -------
    mean_spectrum : np.ndarray, shape (n_features,)
        Mean of the input data across samples.

    lower_bound : np.ndarray, shape (n_features,)
        Lower bound of the computed confidence interval for each feature.

    upper_bound : np.ndarray, shape (n_features,)
        Upper bound of the computed confidence interval for each feature.
    """
    bootstrapped_means = np.array([
        np.nanmean(resample(data, replace=True, n_samples=len(data)), axis=0)
        for _ in range(n_bootstrap)
    ])
    lower_bound = np.percentile(bootstrapped_means, (100 - ci) / 2, axis=0)
    upper_bound = np.percentile(bootstrapped_means, 100 - (100 - ci) / 2, axis=0)
    mean_spectrum = np.nanmean(data, axis=0)
    return mean_spectrum, lower_bound, upper_bound
