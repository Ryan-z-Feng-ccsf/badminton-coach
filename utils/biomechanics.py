import numpy as np

def calculate_3d_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    param a: 3D coordinates of the first point (np.array)
    param b: 3D coordinates of the second point (np.array)
    param c: 3D coordinates of the third point (np.array)
    return: angle in degrees between the vectors AB and bc
    """
    # Create vectors AB and bc
    ab = a - b
    bc = c - b

    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.sum(ab * bc)

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
    return round(float(angle_degrees),3)


def calculate_joint_velocity(
                             vx: list[np.ndarray],
                             vy: list[np.ndarray],
                             vz: list[np.ndarray]
                             ) -> list:
    """
    Calculates the 3D scalar speed from independent X, Y, Z velocity components.

    Args:
        vx: 1D array of velocity components along the X axis.
        vy: 1D array of velocity components along the Y axis.
        vz: 1D array of velocity components along the Z axis.

    Returns:
        list[float]: A list of scalar speeds for each frame, rounded to 3 decimal places.
    """
    vel_vectors = np.stack([vx, vy, vz], axis=1)
    scalar_speeds = np.linalg.norm(vel_vectors, axis=1)
    return np.round(scalar_speeds, 3).tolist()


if __name__ == "__main__":
    # Example usage
    a = np.array([1, 0, 0])
    b = np.array([0, 0, 0])
    c = np.array([0, 1, 0])
    angle = calculate_3d_angle(a, b, c)
    print(f"Angle between AB and BC: {angle} degrees")

