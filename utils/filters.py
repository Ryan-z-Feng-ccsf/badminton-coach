from scipy.signal import savgol_filter
import numpy as np
"""
input:
x :list[float]
y :list[float]
z:list[float]

output:
smoothed_joint_velocity=[...], # list of smoothed shoulder velocities across frames
"""


def calculate_smoothed_vel(
        coordinates: list,
        fps: float,
        window_length: int = 9,
        polyorder: int = 2,
        deriv: int = 1  # Time difference
)->np.ndarray:
    """
    Applies a Savitzky-Golay filter to smooth a 1D time series and calculate its derivative.

    Args:
        data_1d: Raw 1D coordinate array (e.g., just the X axis over time).
        window_length: The length of the filter window (must be an odd integer).
        polyorder: The order of the polynomial used to fit the samples.
        deriv: The order of the derivative to compute (1 for First Derivative).
        delta: The spacing of the samples to which the filter will be applied.

    Returns:
        np.ndarray: The smoothed derivative array (representing 1D velocity).
    """
    delta = 1 / fps
    return savgol_filter(coordinates, window_length, polyorder, deriv, delta)


