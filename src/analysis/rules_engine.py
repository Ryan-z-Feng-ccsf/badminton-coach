from typing import Dict, Any
from biomechanics import calculate_3d_angle, calculate_joint_velocity

import numpy as np


class SafetyRulesLayer:
    """Layer 1: Universal Safety Rules"""

    def __init__(self):
        self.ELBOW_ANGLE_THRESHOLD = 175.0  # Example threshold for elbow angle
        self.SHOULDER_ANGLE_THRESHOLD = 100.0  # Example threshold for shoulder angle

    def check_elbow_hyperextension(self, elbow_angle: float) -> Dict[str, Any]:
        """
        Check if the elbow angle exceeds the hyperextension threshold.
        param elbow_angle: The calculated angle of the elbow joint in degrees
        return: A dictionary containing the diagnosis result, including whether it's safe and any relevant details.
        """
        if elbow_angle > self.ELBOW_ANGLE_THRESHOLD:
            return {
                "issue": "Elbow Hyperextension",
                "is_safe": False,
                "details": f"Elbow angle of {elbow_angle:.2f}°.",
                "threshold": self.ELBOW_ANGLE_THRESHOLD
            }
        if np.isnan(elbow_angle):
            return {
                "issue": "Joint overlap, unable to calculate elbow angle",
            }
        return {
            "issue": "Elbow angle is within safe limits",
            "is_safe": True,
            "details": f"Elbow angle of {elbow_angle:.2f}°.",
            "threshold": self.ELBOW_ANGLE_THRESHOLD
        }

    def check_shoulder_impingement(self, shoulder_angle: float) -> Dict[str, Any]:
        """
        Check if the shoulder angle exceeds the impingement risk threshold.
        param shoulder_angle: The calculated angle of the shoulder joint in degrees
        return: A dictionary containing the diagnosis result, including whether it's safe and any relevant details.
        """
        if shoulder_angle > self.SHOULDER_ANGLE_THRESHOLD:
            return {
                "issue": "Shoulder Impingement Risk",
                "is_safe": False,
                "details": f"Shoulder angle of {shoulder_angle:.2f}°.",
                "threshold": self.SHOULDER_ANGLE_THRESHOLD
            }
        if np.isnan(shoulder_angle):
            return {
                "issue": "Joint overlap, unable to calculate shoulder angle",
            }
        return {
            "issue": "Shoulder angle is within safe limits",
            "is_safe": True,
            "details": f"Shoulder angle of {shoulder_angle:.2f}°.",
            "threshold": self.SHOULDER_ANGLE_THRESHOLD
        }


class TechniqueRulesLayer:
    """Layer 2: Technique-Specific Rules"""

    def __init__(self, length_wrist: float, length_elbow: float, length_shoulder: float, impact: bool = False):
        self.HITTING_HEIGHT_THRESHOLD = length_shoulder + length_elbow + length_wrist  # Example threshold for hitting height in meters
        self.IMPACT = impact  # Flag to indicate if the impact point has been evaluated

    def check_kinetic_chain(self, shoulder_velocity: list[float], elbow_velocity: list[float],
                            wrist_velocity: list[float]) -> Dict[str, Any]:
        """
        check the kinetic chain sequence
        param shoulder_velocity: The velocity of the shoulder joint in m/s
        param elbow_velocity: The velocity of the elbow joint in m/s
        param wrist_velocity: The velocity of the wrist joint in m/s
        return: A dictionary containing the diagnosis result, including whether the kinetic chain is functioning properly and any relevant details.
        """
        if not shoulder_velocity or not elbow_velocity or not wrist_velocity:
            return {
                "issue": "Insufficient data to evaluate kinetic chain",
                "is_proper": False,
                "details": "No velocity data provided for shoulder, elbow, or wrist."
            }
        shoulder_velocity_nd_array = np.asarray(shoulder_velocity)
        elbow_velocity_nd_array = np.asarray(elbow_velocity)
        wrist_velocity_nd_array = np.asarray(wrist_velocity)
        idx_shoulder_peak = int(
            np.argmax(shoulder_velocity_nd_array))  # Find the index of the peak velocity for the shoulder
        idx_elbow_peak = int(np.argmax(elbow_velocity_nd_array))  # Find the index of the peak velocity for the elbow
        idx_wrist_peak = int(np.argmax(wrist_velocity_nd_array))  # Find the index of the peak velocity for the wrist
        if idx_shoulder_peak < idx_elbow_peak < idx_wrist_peak:
            return {
                "issue": "Kinetic chain is functioning properly",
                "is_proper": True,
                "idx_shoulder_peak": idx_shoulder_peak,
                "idx_elbow_peak": idx_elbow_peak,
                "idx_wrist_peak": idx_wrist_peak
            }
        else:
            return {
                "issue": "Kinetic chain is not functioning properly",
                "is_proper": False,
                "idx_shoulder_peak": idx_shoulder_peak,
                "idx_elbow_peak": idx_elbow_peak,
                "idx_wrist_peak": idx_wrist_peak
            }

    def evaluate_impact_point(self, impact_height: float) -> Dict[str, Any]:
        """
        Evaluate the impact point of the ball based on the height of the impact.
        param impact_height: The height of the impact point in meters
        return: A dictionary containing the diagnosis result, including whether the impact point is optimal and any relevant details.
        """
        if impact_height >= self.HITTING_HEIGHT_THRESHOLD and self.IMPACT:
            return {
                "issue": "Impact point is too high",
                "is_optimal": False,
                "detail": impact_height,
                "threshold": self.HITTING_HEIGHT_THRESHOLD
            }
        else:
            return {
                "issue": "Impact point is optimal",
                "is_optimal": True,
                "detail": impact_height,
                "threshold": self.HITTING_HEIGHT_THRESHOLD
            }


def extract_body_part(frame: int, pose_data: dict[int:dict[str:dict[str:int]]], body_part: str) -> list[Any]:
    """
    Extract the 3D coordinates of a specific body part from the pose data for a given frame.
    param frame: The index of the frame to extract data from
    param pose_data: The pose data containing 3D coordinates for each body part across frames
    param body_part: The name of the body part to extract (e.g., "right
_shoulder", "right_elbow", "right_wrist", "right_hip")
    """
    return [pose_data[frame][body_part]["x"], pose_data[frame][body_part]["y"], pose_data[frame][body_part]["z"]]


class DiagnosisEngine:
    """Layer 3: Main Diagnosis Engine"""

    def __init__(self, length_shoulder: float, length_elbow: float, length_wrist: float):
        self.safety_rules_layer = SafetyRulesLayer()
        self.technique_rules_layer = TechniqueRulesLayer(length_shoulder, length_elbow, length_wrist, impact=True)

    def analyze_stroke(self, pose_data: dict[int:dict[str:dict[str:int]]], impact_height: float) -> Dict[str, Any]:
        """
        pose_data: {
            0: {
                "right_shoulder": {"x": 0.5, "y": 0.5, "z": 0.5},
                "right_elbow": {"x": 0.5, "y": 0.5, "z": 0.5},
                "right_wrist": {"x": 0.5, "y": 0.5, "z": 0.5},
                ...
                },
            1: {
                "right_shoulder": {"x": 0.5, "y": 0.5, "z": 0.5},
                "right_elbow": {"x": 0.5, "y": 0.5, "z": 0.5},
                "right_wrist": {"x": 0.5, "y": 0.5, "z": 0.5},
                ...
                },
            ...
        }
        Entry point to process pose data, extract features, apply rules, and return JSON report.
        param pose_data: A list of 3D coordinates representing the pose data for the stroke.
        return: A dictionary containing the diagnosis report, including safety and technique assessments.
        """
        # Extract relevant features from pose data
        right_shoulder = []
        right_elbow = []
        right_wrist = []
        right_hip = []
        report: Dict[int, Any] = {}
        for frame in pose_data.keys():
            right_shoulder.append(extract_body_part(frame, pose_data, "right_shoulder"))
            right_elbow.append(extract_body_part(frame, pose_data, "right_elbow"))
            right_wrist.append(extract_body_part(frame, pose_data, "right_wrist"))
            right_hip.append(extract_body_part(frame, pose_data, "right_hip"))

        # Calculate angles and velocities
        for frame in pose_data.keys():  # calculate angles for each frame and apply safety rules
            right_elbow_angle = calculate_3d_angle(right_shoulder, right_elbow, right_wrist)
            right_shoulder_angle = calculate_3d_angle(right_shoulder, right_elbow, right_wrist)
            elbow_hyperextension_result = self.safety_rules_layer.check_elbow_hyperextension(right_elbow_angle)
            shoulder_impingement_result = self.safety_rules_layer.check_shoulder_impingement(right_shoulder_angle)
            report[frame] = {
                frame: {
                    "right_elbow_report": elbow_hyperextension_result,
                    "right_shoulder_report": shoulder_impingement_result,
                }
            }
        for frame in range(len(right_shoulder) - 1):  # calculate velocities for each frame and apply technique rules
            right_shoulder_velocity = calculate_joint_velocity(frame, right_shoulder)
            right_elbow_velocity = calculate_joint_velocity(frame, right_elbow)
            right_wrist_velocity = calculate_joint_velocity(frame, right_wrist)
            kinetic_chain_result = self.technique_rules_layer.check_kinetic_chain(right_shoulder_velocity,
                                                                                  right_elbow_velocity,
                                                                                  right_wrist_velocity)
            report[frame]["kinetic_chain_report"] = kinetic_chain_result

        if self.technique_rules_layer.evaluate_impact_point(impact_height):
            report["impact_point_report"] = self.technique_rules_layer.evaluate_impact_point(impact_height)
        return report


if __name__ == "__main__":
    # Example usage
    pose_data_example = {
        0: {  # 第0帧：引拍阶段（手臂弯曲）
            "right_shoulder": {"x": 0.5, "y": 0.5, "z": 0.5},
            "right_elbow": {"x": 0.55, "y": 0.45, "z": 0.4},
            "right_wrist": {"x": 0.52, "y": 0.55, "z": 0.3},
            "right_hip": {"x": 0.5, "y": 0.8, "z": 0.5},
        },
        1: {  # 第1帧：挥拍中阶段（手臂开始甩出，速度增加）
            "right_shoulder": {"x": 0.51, "y": 0.5, "z": 0.5},  # 肩膀相对固定
            "right_elbow": {"x": 0.6, "y": 0.35, "z": 0.45},
            "right_wrist": {"x": 0.65, "y": 0.3, "z": 0.5},
            "right_hip": {"x": 0.51, "y": 0.8, "z": 0.5},
        },
        2: {  # 第2帧：击球瞬间（手臂完全伸直，速度达到巅峰，且手腕超过肘部）
            "right_shoulder": {"x": 0.52, "y": 0.5, "z": 0.5},
            "right_elbow": {"x": 0.65, "y": 0.2, "z": 0.55},
            "right_wrist": {"x": 0.75, "y": 0.05, "z": 0.6},  # 手腕y最小（最高），x最大（甩得最远）
            "right_hip": {"x": 0.52, "y": 0.8, "z": 0.5},
        }
    }
    impact_height_example = 2.5
    diagnosis_engine = DiagnosisEngine(length_shoulder=0.3, length_elbow=0.3, length_wrist=0.2)
    report = diagnosis_engine.analyze_stroke(pose_data_example, impact_height_example)
    print(report)
