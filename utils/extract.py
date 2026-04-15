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
from .biomechanics import calculate_joint_velocity, calculate_3d_angle


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


def extract_body_parts(pose_data: dict, joint_name: str) -> np.ndarray:
    """
    Extract the 3D coordinates of a specific body part (joint) from the pose data across all frames.
    param pose_data: The pose data containing 3D coordinates for each body part across frames
    param joint_name: The name of the joint
    return: A np_array of 3D coordinates for the specified joint across all frames
    """
    return np.array([pose_data[frame][joint_name] for frame in sorted(pose_data.keys())])


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
        return round(length_elbow + length_wrist,3)

    def extract_arm_extension_length(self, right_shoulder: np.ndarray, right_wrist: np.ndarray) -> float:
        """
        Extract the arm extension length at impact (straight-line distance from shoulder to wrist).
        This is used to compare against the impact_threshold (max arm length) to evaluate arm straightness.
        param right_shoulder: 3D coordinates of the right shoulder joint
        param right_wrist: 3D coordinates of the right wrist joint
        return: arm_extension_length
        """
        return round(float(np.linalg.norm(right_shoulder[self.impact_frame] - right_wrist[self.impact_frame])),3)


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

    def extract_joint_velocity(self, pose_data: dict, joint_name: str) -> list[float]:
        """
        Extract the velocity of a joint across frames during the impact event.
        param joint_name: The name of the joint to extract velocity for (e.g., "right_shoulder", "right_elbow", "right_wrist")
        param pose_data: The pose data containing 3D coordinates for each body part across frames
        return: A list of velocities for the specified joint across frames
        """
        joint_positions: list = extract_body_parts(pose_data, joint_name).tolist()
        velocities: list = []
        for i in range(len(joint_positions) - 1):
            velocity = calculate_joint_velocity(i, joint_positions)
            velocities.append(velocity)
        return velocities


if __name__ == "__main__":
    # test extract body part
    pose_data_payload = {
        0: {
            "right_shoulder": [0.51, 0.50, 0.50],
            "right_elbow": [0.60, 0.35, 0.45],
            "right_wrist": [0.65, 0.30, 0.50]
        },
        1: {
            "right_shoulder": [0.52, 0.50, 0.50],
            "right_elbow": [0.65, 0.20, 0.55],
            "right_wrist": [0.75, 0.05, 0.60]
        },
        2: {"right_shoulder": [0.52, 0.50, 0.51],
            "right_elbow": [0.65, 0.30, 0.55],
            "right_wrist": [0.75, 0.05, 0.60]
            }
    }
    for frame in sorted(pose_data_payload.keys()):
        right_shoulder = extract_body_part(frame, pose_data_payload, "right_shoulder")
        print(right_shoulder)
    print(extract_body_parts(pose_data_payload, "right_shoulder"))
    right_shoulder = extract_body_parts(pose_data_payload, "right_shoulder")
    right_elbow = extract_body_parts(pose_data_payload, "right_elbow")
    right_wrist = extract_body_parts(pose_data_payload, "right_wrist")
    print("-----------")
    # test extract impact threshold
    impact_metrics = ImpactMetrics(2)
    print(impact_metrics.extract_arm_extension_length(right_shoulder, right_wrist))
    print(impact_metrics.extract_arm_extension_length(right_elbow, right_wrist))
    angle_metrics = AngleMetrics()
    print(angle_metrics.extract_b_joint_angle("right_shoulder", "right_elbow", "right_wrist", pose_data_payload))
    velocity_metrics = VelocityMetrics()
    print(velocity_metrics.extract_joint_velocity(pose_data_payload, "right_shoulder"))
