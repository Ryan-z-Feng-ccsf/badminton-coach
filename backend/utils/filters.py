from scipy.signal import savgol_filter
import numpy as np
"""
input:
x :list[float]
y :list[float]
z :list[float]

output:
smoothed_x :list[float]
smoothed_y :list[float]
smoothed_z :list[float]
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
        coordinates: Raw 1D coordinate array (e.g., just the X axis over time).
        window_length: The length of the filter window (must be an odd integer).
        polyorder: The order of the polynomial used to fit the samples.
        deriv: The order of the derivative to compute (1 for First Derivative).

    Returns:
        np.ndarray: The smoothed derivative array (representing 1D velocity).
    """
    delta = 1 / fps
    return savgol_filter(coordinates, window_length, polyorder, deriv, delta) # pyright: ignore[reportReturnType]

if __name__ == "__main__":
    sample_data=[
        0.577,0.574,0.573,0.572,
        0.572,0.57, 0.567,0.565,
        0.565,0.561,0.56,0.558
                 ]
    print(calculate_smoothed_vel(sample_data, fps=60))
