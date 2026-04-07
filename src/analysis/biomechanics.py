import numpy as np


def calculate_3d_angle(a: list[float], b: list[float], c: list[float]) -> float:
    """
    param a: 3D coordinates of the first point (list)
    param b: 3D coordinates of the second point (list)
    param c: 3D coordinates of the third point (list)
    return: angle in degrees between the vectors AB and bc
    """
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    # Create vectors AB and bc
    ab = a - b
    bc = c - b

    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(ab, bc)

    magnitude_ab = np.linalg.norm(ab)
    magnitude_bc = np.linalg.norm(bc)
    if magnitude_ab < 1e-6 or magnitude_bc < 1e-6:
        return np.nan
    cos_angle = dot_product / (magnitude_ab * magnitude_bc)

    # Ensure the cosine value is within the valid range [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    # Calculate the angle in radians
    angle = np.arccos(cos_angle)
    # Convert the angle to degrees
    angle_degrees = np.degrees(angle)
    return angle_degrees
