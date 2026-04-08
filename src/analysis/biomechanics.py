import numpy as np


def calculate_3d_angle(a: list[float], b: list[float], c: list[float]) -> float:
    """
    param a: 3D coordinates of the first point (np.array)
    param b: 3D coordinates of the second point (np.array)
    param c: 3D coordinates of the third point (np.array)
    return: angle in degrees between the vectors AB and bc
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
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
    return angle_degrees


def calculate_joint_velocity(i: int, joint: list[np.ndarray]) -> float:
    """
    Calculate the velocity of a joint between two consecutive frames.
    param i: The index of the current frame
    param joint: A list of 3D coordinates for the joint across frames
    return: The velocity of the joint in m/s, calculated as the distance between the joint positions in consecutive frames divided by the time interval (assuming 30 FPS, so time interval is 1/30 seconds).
    """
    dx = joint[i + 1][0] - joint[i][0]
    dy = joint[i + 1][1] - joint[i][1]
    dz = joint[i + 1][2] - joint[i][2]
    shoulder_velocity = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return shoulder_velocity


if __name__ == "__main__":
    # Example usage
    a = [1, 0, 0]
    b = [0, 0, 0]
    c = [0, 1, 0]
    angle = calculate_3d_angle(a, b, c)
    print(f"Angle between AB and BC: {angle} degrees")

    joint_positions = [np.array([0, 0, 0]), np.array([1, 1, 1])]
    velocity = calculate_joint_velocity(0, joint_positions)
    print(f"Joint velocity: {velocity} m/s")
