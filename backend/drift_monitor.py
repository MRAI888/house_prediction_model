import numpy as np
import pandas as pd

def calculate_psi(expected, actual, buckets=10):
    """Population Stability Index."""
    breaks = np.percentile(expected, np.linspace(0, 100, buckets+1))
    expected_percents = np.histogram(expected, breaks)[0] / len(expected)
    actual_percents = np.histogram(actual, breaks)[0] / len(actual)

    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 1e-10, expected_percents)
    actual_percents = np.where(actual_percents == 0, 1e-10, actual_percents)

    psi = np.sum((actual_percents - expected_percents) *
                 np.log(actual_percents / expected_percents))
    return psi

def check_drift(reference_df, current_df, features, threshold=0.2):
    """Return PSI for each feature and alert flag."""
    drift_report = {}
    for feat in features:
        psi = calculate_psi(reference_df[feat].dropna(),
                            current_df[feat].dropna())
        drift_report[feat] = {'PSI': round(psi, 4),
                              'Alert': psi > threshold}
    return drift_report