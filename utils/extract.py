"""
input:
metadata=
{
fps=?

{pose_data_payload = {
    0: {
        "right_shoulder": [0.51, 0.50, 0.50], # [x, y, z]
        "right_elbow": [0.60, 0.35, 0.45],
        "right_wrist": [0.65, 0.30, 0.50]
    },
    1: {
        "right_shoulder": [0.52, 0.50, 0.50],
        "right_elbow": [0.65, 0.20, 0.55],
        "right_wrist": [0.75, 0.05, 0.60]
    },
    # ... 后续的 2, 3, 4 帧
}}}
output:
impact_height : float
length_elbow : float
length_wrist : float
right_elbow_angle : list[float]
right_shoulder_angle : list[float]
"""
import numpy as np
from biomechanics import calculate_joint_velocity, calculate_3d_angle


def extract_body_part(frame: int, pose_data: dict, joint_name: str) -> np.ndarray:
    """
    Extract the 3D coordinates of a specific body part (joint) from the pose data for a given frame.
    param frame: The index of the frame to extract the body part from
    param pose_data: The pose data containing 3D coordinates for each body part across frames
    param joint_name: The name of the joint to extract (e.g., "right_shoulder", "right_elbow", "right_wrist")
    return: A np_array of 3D coordinates for the specified joint in the given frame
    """
    coords = pose_data[frame][joint_name]
    return np.array(coords)


class ImpactMetrics:
    """
    A class to extract key biomechanical metrics related to the impact event,
    such as the impact height and the impact threshold
    """

    def __init__(self, impact_frame: int):
        self.impact_frame = impact_frame

    def extract_impact_threshold(self, right_shoulder: np.ndarray, right_elbow: np.ndarray,
                                 right_wrist: np.ndarray) -> float:
        """
        Extract key biomechanical metrics related to the impact event, such as the height of the impact, the lengths of the upper arm and forearm, and the angles at the elbow and shoulder joints.
        param right_shoulder: 3D coordinates of the right shoulder joint
        param right_elbow: 3D coordinates of the right elbow joint
        param right_wrist: 3D coordinates of the right wrist joint
        return: impact_threshold
        """

        length_wrist = float(np.linalg.norm(right_wrist[self.impact_frame] - right_elbow[self.impact_frame]))
        length_elbow = float(np.linalg.norm(right_elbow[self.impact_frame] - right_shoulder[self.impact_frame]))
        return length_elbow + length_wrist

    def extract_impact_height(self, right_shoulder: np.ndarray, right_wrist: np.ndarray) -> float:
        """
        Extract the height of the impact event, which can be defined as the vertical distance from the ground to the point of impact (e.g., the wrist).
        param right_shoulder: 3D coordinates of the right shoulder joint
        param right_wrist: 3D coordinates of the right wrist joint
        return: impact_height
        """
        return float(np.linalg.norm(right_shoulder[self.impact_frame] - right_wrist[self.impact_frame]))


class AngleMetrics:
    """
    A class to extract key biomechanical metrics related to the angles
    at the elbow and shoulder joints during the impact event.
    """

    def extract_b_joint_angle(self, a_joint_name: str, b_joint_name: str, c_joint_name: str,
                              pose_data: dict) -> list[float]:
        """
        Extract the angles at the elbow and shoulder joints during the impact event.
        param a_joint_name: The name of the first joint (e.g., "right_shoulder")
        param b_joint_name: The name of the second joint (e.g., "right_elbow")
        param c_joint_name: The name of the third joint (e.g., "right_wrist")
        param pose_data: The pose data containing 3D coordinates for each body part across frames
        return: A list of angles for the specified joint across frames
        """
        angles = []
        for frame in sorted(pose_data.keys()):
            a = extract_body_part(frame, pose_data, a_joint_name)
            b = extract_body_part(frame, pose_data, b_joint_name)
            c = extract_body_part(frame, pose_data, c_joint_name)
            angle = calculate_3d_angle(a, b, c)
            angles.append(angle)
        return angles


class VelocityMetrics:
    """
    A class to extract key biomechanical metrics related to the velocity of the joints during the impact event.
    """

    def extract_joint_velocity(self, joint_name: str, pose_data: dict) -> list[float]:
        """
        Extract the velocity of a joint across frames during the impact event.
        param joint_name: The name of the joint to extract velocity for (e.g., "right_shoulder", "right_elbow", "right_wrist")
        param pose_data: The pose data containing 3D coordinates for each body part across frames
        return: A list of velocities for the specified joint across frames
        """
        joint_positions: list = []
        for frame in sorted(pose_data.keys()):
            joint_position = extract_body_part(frame, pose_data, joint_name)
            joint_positions.append(joint_position)
        velocities: list = []
        for i in range(len(joint_position)):
            velocity = calculate_joint_velocity(i, joint_positions)
            velocities.append(velocity)
        return velocities


if __name__ == "__main__":
    pass