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
smooth_shoulder_velocity: list[float]
smooth_elbow_velocity: list[float]
smooth_wrist_velocity: list[float]
"""
import numpy as np
from biomechanics import calculate_joint_velocity, calculate_3d_angle
from filters import calculate_smoothed_vel


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

def extract_coords(pose_data:dict,joint_name:str,coords_idx:int=0)->list:
    """
    Extract the 1D coordinates
    """
    return  [pose_data[frame][joint_name][coords_idx] for frame in sorted(pose_data.keys())]

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

    def extract_joint_velocity(self, pose_data: dict, joint_name: str,fps:float) -> list[float]:
        """
        Extract the velocity of a joint across frames during the impact event.
        param joint_name: The name of the joint to extract velocity for (e.g., "right_shoulder", "right_elbow", "right_wrist")
        param pose_data: The pose data containing 3D coordinates for each body part across frames
        return: A list of velocities for the specified joint across frames
        """
        joint_positions_x: list = extract_coords(pose_data,joint_name,0)
        joint_positions_y: list = extract_coords(pose_data,joint_name,1)
        joint_positions_z: list = extract_coords(pose_data,joint_name,2)
        vx = calculate_smoothed_vel(joint_positions_x,fps)
        vy = calculate_smoothed_vel(joint_positions_y,fps)
        vz = calculate_smoothed_vel(joint_positions_z,fps)
        velocities:list = calculate_joint_velocity(vx, vy, vz)

        return velocities


if __name__ == "__main__":
    # test extract body part
    pose_data_payload = {
        0: {'right_shoulder': [0.577, 0.448, -0.005], 'right_elbow': [0.621, 0.497, -0.044], 'right_wrist': [0.623, 0.415, -0.063], 'right_hip': [0.577, 0.643, -0.02]},
        1: {'right_shoulder': [0.574, 0.448, 0.01], 'right_elbow': [0.62, 0.497, -0.022], 'right_wrist': [0.621, 0.413, -0.031], 'right_hip': [0.576, 0.642, -0.017]},
        2: {'right_shoulder': [0.573, 0.447, 0.016], 'right_elbow': [0.619, 0.492, 0.004], 'right_wrist': [0.615, 0.405, 0.017], 'right_hip': [0.574, 0.641, -0.014]},
        3: {'right_shoulder': [0.572, 0.447, 0.014], 'right_elbow': [0.619, 0.489, -0.047], 'right_wrist': [0.612, 0.402, -0.116], 'right_hip': [0.573, 0.641, -0.013]},
        4: {'right_shoulder': [0.572, 0.447, 0.013], 'right_elbow': [0.619, 0.488, -0.084], 'right_wrist': [0.611, 0.4, -0.175], 'right_hip': [0.573, 0.641, -0.013]},
        5: {'right_shoulder': [0.57, 0.444, 0.012], 'right_elbow': [0.619, 0.476, -0.076], 'right_wrist': [0.604, 0.387, -0.118], 'right_hip': [0.57, 0.638, -0.012]},
        6: {'right_shoulder': [0.567, 0.437, 0.012], 'right_elbow': [0.618, 0.455, -0.018], 'right_wrist': [0.591, 0.369, 0.008], 'right_hip': [0.568, 0.632, -0.011]},
        7: {'right_shoulder': [0.565, 0.433, 0.011], 'right_elbow': [0.617, 0.449, -0.022], 'right_wrist': [0.589, 0.365, -0.026], 'right_hip': [0.567, 0.63, -0.008]},
        8: {'right_shoulder': [0.565, 0.432, 0.01], 'right_elbow': [0.616, 0.446, -0.012], 'right_wrist': [0.588, 0.364, -0.009], 'right_hip': [0.566, 0.628, -0.007]},
        9: {'right_shoulder': [0.561, 0.425, 0.011], 'right_elbow': [0.614, 0.422, 0.034], 'right_wrist': [0.577, 0.351, 0.092], 'right_hip': [0.565, 0.623, -0.005]},
        10: {'right_shoulder': [0.56, 0.421, 0.018], 'right_elbow': [0.613, 0.416, 0.064], 'right_wrist': [0.575, 0.349, 0.14], 'right_hip': [0.565, 0.621, -0.004]},
        11: {'right_shoulder': [0.558, 0.409, 0.02], 'right_elbow': [0.608, 0.383, 0.094], 'right_wrist': [0.574, 0.341, 0.215], 'right_hip': [0.564, 0.62, -0.003]},
        12: {'right_shoulder': [0.557, 0.405, 0.022], 'right_elbow': [0.606, 0.376, 0.095], 'right_wrist': [0.574, 0.34, 0.211], 'right_hip': [0.564, 0.62, -0.001]},
        13: {'right_shoulder': [0.555, 0.392, 0.031], 'right_elbow': [0.594, 0.349, 0.131], 'right_wrist': [0.579, 0.32, 0.282], 'right_hip': [0.567, 0.616, 0.002]},
        14: {'right_shoulder': [0.554, 0.384, 0.026], 'right_elbow': [0.577, 0.32, 0.105], 'right_wrist': [0.579, 0.272, 0.26], 'right_hip': [0.572, 0.614, 0.001]},
        15: {'right_shoulder': [0.554, 0.382, 0.03], 'right_elbow': [0.571, 0.318, 0.121], 'right_wrist': [0.579, 0.266, 0.285], 'right_hip': [0.574, 0.614, -0.0]},
        16: {'right_shoulder': [0.554, 0.38, 0.027], 'right_elbow': [0.571, 0.316, 0.114], 'right_wrist': [0.579, 0.263, 0.272], 'right_hip': [0.575, 0.613, -0.001]},
        17: {'right_shoulder': [0.555, 0.375, 0.027], 'right_elbow': [0.563, 0.289, 0.1], 'right_wrist': [0.565, 0.201, 0.235], 'right_hip': [0.58, 0.611, 0.005]},
        18: {'right_shoulder': [0.556, 0.378, 0.026], 'right_elbow': [0.559, 0.288, 0.092], 'right_wrist': [0.55, 0.217, 0.217], 'right_hip': [0.586, 0.613, 0.007]},
        19: {'right_shoulder': [0.556, 0.378, 0.021], 'right_elbow': [0.558, 0.288, 0.082], 'right_wrist': [0.545, 0.219, 0.203], 'right_hip': [0.587, 0.614, 0.01]}
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
    print(velocity_metrics.extract_joint_velocity(pose_data_payload, "right_shoulder",fps=60))
