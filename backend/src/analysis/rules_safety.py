from abc import ABC, abstractmethod
import numpy as np


class AbstractRuleLayer(ABC):
    @abstractmethod
    def evaluate(self) -> dict:
        """the base for all the technique layer

        Args:
            pose_data_payload (dict): pose data

        Returns:
            dict: Respetive report
        """
        pass


class SafetyRulesCenter:
    def check_max_extension(
        self, 
        joint_name: str, 
        extract_joint_list:dict,  
        safe_threshold: float
    ) -> dict:
        """Check if the angle exceeds the threshold

        Args:
            joint_name (str): _description_
            curr_max_angle (float): _description_
            safe_threshold (float): _description_

        Returns:
            dict: _description_
        """
        curr_max_angle: float = max(extract_joint_list[joint_name])
        if np.isnan(curr_max_angle):
            return {
                "issue": f"Joint overlap, unable to calculate {joint_name} angle",
                "is_safe": None,
                "max_elbow_angle": None,
            }
        if curr_max_angle > safe_threshold:
            return {
                "issue": f"{joint_name} Hyperextension/Impingement Risk",
                "is_safe": False,
                "max_angle": curr_max_angle,
            }
        return {
            "issue": f"{joint_name} angle is within safe limits",
            "is_safe": True,
            "max_angle": curr_max_angle,
        }
        
    def check_min_flexion(
        self, 
        joint_name: str, 
        extract_joint_list:dict,  
        safe_threshold: float
    ) -> dict:
        """Check if the angle exceeds the threshold

        Args:
            joint_name (str): _description_
            curr_max_angle (float): _description_
            safe_threshold (float): _description_

        Returns:
            dict: _description_
        """
        curr_max_angle: float = min(extract_joint_list[joint_name])
        if np.isnan(curr_max_angle):
            return {
                "issue": f"Joint overlap, unable to calculate {joint_name} angle",
                "is_safe": None,
                "max_elbow_angle": None,
            }
        if curr_max_angle > safe_threshold:
            return {
                "issue": f"{joint_name} Hyperextension/Impingement Risk",
                "is_safe": False,
                "max_angle": curr_max_angle,
            }
        return {
            "issue": f"{joint_name} angle is within safe limits",
            "is_safe": True,
            "max_angle": curr_max_angle,
        }