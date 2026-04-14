from typing import Dict, Any
from biomechanics import calculate_3d_angle, calculate_joint_velocity

import numpy as np

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
{'safety_report': 
{
'elbow_hyperextension': 
{'issue': 'Elbow angle is within safe limits', 'is_safe': True, 'max_elbow_angle': 167.8816167125607}, 
'shoulder_impingement': 
{'issue': 'Shoulder Impingement Risk', 'is_safe': False, 'max_shoulder_angle': 155.09551895544325}
}, 
'technique_report': 
{
'kinetic_chain': 
{'issue': 'Kinetic chain is not functioning properly', 'is_proper': False, 'idx_shoulder_peak': 0, 'idx_elbow_peak': 1, 'idx_wrist_peak': 0}, 
'impact_point': {'issue': 'Arm fully locked at impact, high risk of injury', 'is_optimal': True, 'impact_height': 2.5, 'threshold': 0.8}
}
}

"""


class SafetyRulesLayer:
    """Layer 1: Universal Safety Rules"""

    def __init__(self):
        self.ELBOW_ANGLE_THRESHOLD = 175.0  # Example threshold for elbow angle
        self.SHOULDER_ANGLE_THRESHOLD = 100.0  # Example threshold for shoulder angle

    def check_elbow_hyperextension(self, elbow_angle_max: float) -> Dict[str, Any]:
        """
        Check if the elbow angle exceeds the hyperextension threshold.
        param elbow_angle: The calculated angle of the elbow joint in degrees
        return: A dictionary containing the diagnosis result, including whether it's safe and any relevant details.
        """
        if elbow_angle_max > self.ELBOW_ANGLE_THRESHOLD:
            return {
                "issue": "Elbow Hyperextension Risk",
                "is_safe": False,
                "max_elbow_angle": elbow_angle_max
            }
        if np.isnan(elbow_angle_max):
            return {
                "issue": "Joint overlap, unable to calculate elbow angle",
                "is_safe": False,
                "max_elbow_angle": None
            }
        return {
            "issue": "Elbow angle is within safe limits",
            "is_safe": True,
            "max_elbow_angle": elbow_angle_max
        }

    def check_shoulder_impingement(self, shoulder_angle_max: float) -> Dict[str, Any]:
        """
        Check if the shoulder angle exceeds the impingement risk threshold.
        param shoulder_angle: The calculated angle of the shoulder joint in degrees
        return: A dictionary containing the diagnosis result, including whether it's safe and any relevant details.
        """
        if shoulder_angle_max > self.SHOULDER_ANGLE_THRESHOLD:
            return {
                "issue": "Shoulder Impingement Risk",
                "is_safe": False,
                "max_shoulder_angle": shoulder_angle_max
            }
        if np.isnan(shoulder_angle_max):
            return {
                "issue": "Joint overlap, unable to calculate shoulder angle",
                "is_safe": False,
                "max_shoulder_angle": None
            }
        return {
            "issue": "Shoulder angle is within safe limits",
            "is_safe": True,
            "max_shoulder_angle": shoulder_angle_max
        }


class TechniqueRulesLayer:
    """Layer 2: Technique-Specific Rules"""

    def __init__(self, length_wrist: float, length_elbow: float, length_shoulder: float, fps: float,
                 impact: bool = False, lock_seconds: float = 0.4):
        self.HITTING_HEIGHT_THRESHOLD = length_shoulder + length_elbow + length_wrist  # Example threshold for hitting height in meters
        self.IMPACT = impact  # Flag to indicate if the impact point has been evaluated
        # Set up rules for evaluating the impact point, such as optimal height range.
        # format: (min_ratio, max_ratio, is_optimal, message)
        self.IMPACT_RULES = [
            (0.0, 0.80, False, "Impact point is too low, likely hitting the net or causing a weak shot"),
            (0.80, 0.95, True, "Impact point is optimal, allowing for good power and control"),
            (0.95, float('inf'), False, "Arm fully locked at impact, high risk of injury")
        ]
        self.window_frame = int(round(fps * lock_seconds))

    def check_kinetic_chain(self, smoothed_right_shoulder_velocity: list[float],
                            smoothed_right_elbow_velocity: list[float],
                            smoothed_right_wrist_velocity: list[float], impact_frame: float) -> Dict[str, Any]:
        """
        check the kinetic chain sequence
        param shoulder_velocity: The smoothed velocity of the shoulder joint in m/s
        param elbow_velocity: The smoothed velocity of the elbow joint in m/s
        param wrist_velocity: The smoothed velocity of the wrist joint in m/s
        param impact_frame: The index of the frame where the impact occurs, used to focus the kinetic chain analysis around this point
        return: A dictionary containing the diagnosis result, including whether the kinetic chain is functioning properly and any relevant details.
        """
        if not smoothed_right_shoulder_velocity or not smoothed_right_elbow_velocity or not smoothed_right_wrist_velocity:
            return {
                "issue": "Insufficient data to evaluate kinetic chain",
                "is_proper": False,
                "idx_shoulder_peak": None,
                "idx_elbow_peak": None,
                "idx_wrist_peak": None
            }
        # if impact_frame is less than the window frame, we can only analyze from the start of the data to the impact frame
        start_frame = max(0, int(impact_frame) - int(self.window_frame))  # Analyze a window of frames leading up to the impact frame to check the sequence of velocity peaks
        end_frame = impact_frame + 1
        shoulder_slice = np.asarray(smoothed_right_shoulder_velocity[start_frame:end_frame])
        elbow_slice = np.asarray(smoothed_right_elbow_velocity[start_frame:end_frame])
        wrist_slice = np.asarray(smoothed_right_wrist_velocity[start_frame:end_frame])
        # if shoulder or elbow or wrist velocity data is empty in the analyzed window, return insufficient data
        if len(shoulder_slice) == 0 or len(elbow_slice) == 0 or len(wrist_slice) == 0:
            return {
                "issue": "Velocity slices are empty, check impact_frame validity",
                "is_proper": False,
                "idx_shoulder_peak": None,
                "idx_elbow_peak": None,
                "idx_wrist_peak": None
            }
        idx_shoulder_peak = int(
            np.argmax(
                shoulder_slice)) + start_frame  # Find the index of the peak velocity for the shoulder, and adjust it to the original frame index by adding start_frame
        idx_elbow_peak = int(
            np.argmax(elbow_slice)) + start_frame  # Find the index of the peak velocity for the elbow
        idx_wrist_peak = int(
            np.argmax(wrist_slice)) + start_frame  # Find the index of the peak velocity for the wrist
        if idx_shoulder_peak <= idx_elbow_peak <= idx_wrist_peak:
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
        if not self.IMPACT:
            return {
                "issue": "Impact point evaluation not performed",
                "is_optimal": None,
                "impact_height": impact_height,
                "threshold": self.HITTING_HEIGHT_THRESHOLD
            }

        impact_ratio = impact_height / self.HITTING_HEIGHT_THRESHOLD
        for min_ratio, max_ratio, is_optimal, message in self.IMPACT_RULES:
            if min_ratio <= impact_ratio < max_ratio:
                return {
                    "issue": message,
                    "is_optimal": is_optimal,
                    "impact_height": impact_height,
                    "threshold": self.HITTING_HEIGHT_THRESHOLD
                }
        return {
            "issue": "Impact point evaluation failed, height ratio out of expected range",
            "is_optimal": None,
            "impact_height": impact_height,
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
        report_result: Dict[str, Any] = {}
        for frame in sorted(pose_data.keys()):
            right_shoulder.append(extract_body_part(frame, pose_data, "right_shoulder"))
            right_elbow.append(extract_body_part(frame, pose_data, "right_elbow"))
            right_wrist.append(extract_body_part(frame, pose_data, "right_wrist"))
            right_hip.append(extract_body_part(frame, pose_data, "right_hip"))

        right_shoulder_angle = []
        right_elbow_angle = []

        # Calculate angles and velocities
        for frame in sorted(pose_data.keys()):  # calculate angles for each frame and apply safety rules
            right_elbow_angle.append(calculate_3d_angle(right_shoulder[frame], right_elbow[frame], right_wrist[frame]))
            right_shoulder_angle.append(
                calculate_3d_angle(right_elbow[frame], right_shoulder[frame], right_hip[frame]))

        elbow_hyperextension_result = self.safety_rules_layer.check_elbow_hyperextension(max(right_elbow_angle))
        shoulder_impingement_result = self.safety_rules_layer.check_shoulder_impingement(max(right_shoulder_angle))
        report_result["safety_report"] = {
            "elbow_hyperextension": elbow_hyperextension_result,
            "shoulder_impingement": shoulder_impingement_result
        }

        right_shoulder_velocity = []
        right_elbow_velocity = []
        right_wrist_velocity = []
        for frame in range(len(right_shoulder) - 1):  # calculate velocities for each frame and apply technique rules
            right_shoulder_velocity.append(calculate_joint_velocity(frame, right_shoulder))
            right_elbow_velocity.append(calculate_joint_velocity(frame, right_elbow))
            right_wrist_velocity.append(calculate_joint_velocity(frame, right_wrist))

        if self.technique_rules_layer.evaluate_impact_point(impact_height):
            report_result["technique_report"] = {
                "kinetic_chain": self.technique_rules_layer.check_kinetic_chain(right_shoulder_velocity,
                                                                                right_elbow_velocity,
                                                                                right_wrist_velocity),
                "impact_point": self.technique_rules_layer.evaluate_impact_point(impact_height)}
        return report_result


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
