from typing import Dict, Any
import numpy as np
import math
"""
input:
metadata=
{
fps=?
smoothed_right_shoulder_velocity=[...], # list of smoothed shoulder velocities across frames
smoothed_right_elbow_velocity=[...], # list of smoothed elbow velocities across frames
smoothed_right_wrist_velocity=[...], # list of smoothed wrist velocities across frames
impact_height=?
length_elbow=?
length_wrist=?
right_elbow_angle=[...], # list of calculated elbow angles across frames
right_shoulder_angle=[...], # list of calculated shoulder angles across frames
}
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
        self._ELBOW_ANGLE_THRESHOLD = 175.0  # Example threshold for elbow angle
        self._SHOULDER_ANGLE_THRESHOLD = 100.0  # Example threshold for shoulder angle

    def check_elbow_hyperextension(self, elbow_angle_max: float) -> Dict[str, Any]:
        """
        Check if the elbow angle exceeds the hyperextension threshold.
        param elbow_angle: The calculated angle of the elbow joint in degrees
        return: A dictionary containing the diagnosis result, including whether it's safe and any relevant details.
        """
        if elbow_angle_max > self._ELBOW_ANGLE_THRESHOLD:
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
        if shoulder_angle_max > self._SHOULDER_ANGLE_THRESHOLD:
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

    def __init__(self, impact_threshold: float, arm_extension_length: float, fps: float, impact_frame: float,
                 lock_seconds: float = 0.4):

        # lock_seconds: the swing from start to finish

        self._HITTING_HEIGHT_THRESHOLD = impact_threshold  # Example threshold for hitting height in meters
        # Set up rules for evaluating the impact point, such as optimal height range.
        # format: (min_ratio, max_ratio, is_optimal, message)
        self._IMPACT_RULES = [
            (0.0, 0.80, False, "Impact point is too low, likely hitting the net or causing a weak shot"),
            (0.80, 0.95, True, "Impact point is optimal, allowing for good power and control"),
            (0.95, float('inf'), False, "Arm fully locked at impact, high risk of injury")
        ]
        self._impact_frame = impact_frame
        self._window_frame = int(round(fps * lock_seconds))
        self._arm_extension_length = arm_extension_length
        self._IMPACT_TOLERANCE :int = math.ceil(10 / 1000 * fps)

    def check_kinetic_chain(self, smoothed_right_shoulder_velocity: list[float],
                            smoothed_right_elbow_velocity: list[float],
                            smoothed_right_wrist_velocity: list[float]) -> Dict[str, Any]:
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
        start_frame = max(0, int(self._impact_frame) - int(self._window_frame))  # Analyze a window of frames leading up to the impact frame to check the sequence of velocity peaks
        end_frame = self._impact_frame + 1
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
        if (idx_shoulder_peak - self._IMPACT_TOLERANCE) <= idx_elbow_peak and (idx_elbow_peak - self._IMPACT_TOLERANCE) <= idx_wrist_peak:
            return {
                "impact_tolerance": self._IMPACT_TOLERANCE,
                "issue": "Kinetic chain is functioning properly",
                "is_proper": True,
                "idx_shoulder_peak": idx_shoulder_peak,
                "idx_elbow_peak": idx_elbow_peak,
                "idx_wrist_peak": idx_wrist_peak
            }
        else:
            return {
                "impact_tolerance": self._IMPACT_TOLERANCE,
                "issue": "Kinetic chain is not functioning properly",
                "is_proper": False,
                "idx_shoulder_peak": idx_shoulder_peak,
                "idx_elbow_peak": idx_elbow_peak,
                "idx_wrist_peak": idx_wrist_peak
            }

    def evaluate_arm_extension_length(self) -> Dict[str, Any]:
        """
        Evaluate the impact point of the ball based on the height of the impact.
        param impact_height: The height of the impact point in meters
        return: A dictionary containing the diagnosis result, including whether the impact point is optimal and any relevant details.
        """
        impact_ratio = self._arm_extension_length / self._HITTING_HEIGHT_THRESHOLD
        for min_ratio, max_ratio, is_optimal, message in self._IMPACT_RULES:
            if min_ratio <= impact_ratio < max_ratio:
                return {
                    "issue": message,
                    "is_optimal": is_optimal,
                    "impact_height": self._arm_extension_length,
                    "threshold": self._HITTING_HEIGHT_THRESHOLD
                }
        return {
            "issue": "Impact point evaluation failed, height ratio out of expected range",
            "is_optimal": None,
            "impact_height": self._arm_extension_length,
            "threshold": self._HITTING_HEIGHT_THRESHOLD
        }

class DiagnosisEngine:
    """Layer 3: Main Diagnosis Engine"""

    def __init__(self, impact_threshold, arm_extension_length: float, fps:float,  impact_frame: float,):
        self._safety_rules_layer = SafetyRulesLayer()
        self._technique_rules_layer = TechniqueRulesLayer(impact_threshold, arm_extension_length, fps, impact_frame)

    def analyze_stroke(self, smoothed_right_shoulder_velocity: list[float],
                       smoothed_right_elbow_velocity: list[float],
                       smoothed_right_wrist_velocity: list[float],
                       right_shoulder_angle: list[float], right_elbow_angle: list[float],
                       ) -> Dict[str, Any]:
        """
        smoothed_right_shoulder_velocity: A list of smoothed velocities for the right shoulder joint across frames
        smoothed_right_elbow_velocity: A list of smoothed velocities for the right elbow joint across frames
        smoothed_right_wrist_velocity: A list of smoothed velocities for the right wrist joint across frames
        right_elbow_angle: A list of calculated angles for the right elbow joint across frames
        right_shoulder_angle: A list of calculated angles for the right shoulder joint across frames
        impact_height: The height of the impact point in meters
        return: A dictionary containing the diagnosis report, including safety and technique assessments.
        """

        report_result: Dict[str, Any] = {}
        elbow_hyperextension_result = self._safety_rules_layer.check_elbow_hyperextension(max(right_elbow_angle))
        shoulder_impingement_result = self._safety_rules_layer.check_shoulder_impingement(max(right_shoulder_angle))
        report_result["safety_report"] = {
            "elbow_hyperextension": elbow_hyperextension_result,
            "shoulder_impingement": shoulder_impingement_result
        }

        report_result["technique_report"] = {
            "kinetic_chain": self._technique_rules_layer.check_kinetic_chain(smoothed_right_shoulder_velocity,
                                                                             smoothed_right_elbow_velocity,
                                                                             smoothed_right_wrist_velocity),
            "impact_point": self._technique_rules_layer.evaluate_arm_extension_length()}
        return report_result


if __name__ == "__main__":
    pass
