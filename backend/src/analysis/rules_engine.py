from rules_safety import AbstractRuleLayer,SafetyRulesCenter
from rules_dominant import DominantArmRules

class DiagnosisEngine:
    def __init__(self,
                 impact_threshold: float, 
                 arm_extension_length: float, 
                 fps: float, 
                 impact_frame: float,
                 ):
        self._safe_center = SafetyRulesCenter()

        self._rules: list[AbstractRuleLayer] = [
            DominantArmRules(
                impact_threshold, 
                 arm_extension_length, 
                 fps, 
                 impact_frame
                 )
        ]

        self._THRESHOLD = {
            "right_elbow_max": 175.0, 
            "right_shoulder_max": 100.0,
            'right_knee_min': 90.0
            }

    def analyze_stroke(self, extract_joint_list: dict) -> dict:
        report_result: dict = {"safety_report": {}, "technique_report": {}}
        #TODO
        report_result["safety_report"]["shoulder_impingement"]=self._safe_center.check_max_extension(
            'right_shoulder_angle',
            extract_joint_list,
            self._THRESHOLD['right_shoulder_max']
            
        )
        report_result["safety_report"]["elbow_hyperextension"]=self._safe_center.check_max_extension(
            'right_elbow_angle',
            extract_joint_list,
            self._THRESHOLD['right_elbow_max']
            
        )
        report_result['safety_report']['knee_hyperflexion']=self._safe_center.check_min_flexion(
            'right_knee_angle',
            extract_joint_list,
            self._THRESHOLD['right_knee_min']
        )
        for rule in self._rules:
            rule_key = rule.__class__.__name__
            report_result["technique_report"][rule_key] = rule.evaluate(
                extract_joint_list
            )
        return report_result

if __name__ == '__main__':
    diagnose= DiagnosisEngine(2.85,2.60,60.0,20)
    extract_data_list={
    # 静态/单次评估指标 (单位: 米)
    "impact_threshold": 2.85,       # 理想最大击球高度
    "arm_extension_length": 2.60,   # 实际击球时的手臂伸展长度 (占比约 91%，在 optimal 范围内)
    
    # 角度序列 (单位: 度, 模拟从引拍到击球再到收拍的 10 帧过程)
    "right_elbow_angle": [95.2, 110.5, 135.0, 155.5, 168.2, 172.5, 165.0, 150.2, 140.5, 135.0],
    "right_shoulder_angle": [110.0, 125.5, 140.2, 155.8, 162.5, 158.0, 145.5, 130.0, 120.5, 115.0],
    
    # 速度序列 (单位: m/s, 模拟动力链传导)
    # 峰值出现顺序: 肩(idx=2) -> 肘(idx=3) -> 腕(idx=4)
    "right_shoulder_velocity": [2.5, 5.8, 8.5, 6.2, 4.0, 2.5, 1.5, 1.0, 0.8, 0.5],
    "right_elbow_velocity": [1.5, 4.0, 8.2, 14.5, 10.2, 6.5, 3.5, 2.0, 1.2, 0.8],
    "right_wrist_velocity": [1.0, 2.5, 6.0, 16.5, 28.5, 20.0, 12.5, 6.0, 3.0, 1.5]
}
    print(diagnose.analyze_stroke(extract_data_list))