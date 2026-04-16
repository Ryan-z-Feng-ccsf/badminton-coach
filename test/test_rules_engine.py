import pytest
from src.analysis.rules_engine import SafetyRulesLayer, TechniqueRulesLayer, DiagnosisEngine

@pytest.fixture
def sample_pose_data():
    return {
        "shoulder_v": [1.0, 2.0, 3.0, 5.0, 2.0, 1.0, 0.5],
        "elbow_v": [0.5, 1.0, 2.0, 4.0, 8.0, 3.0, 1.0],
        "wrist_v": [0.2, 0.5, 1.0, 2.0, 5.0, 12.0, 4.0],
        # Group 1: Safe stroke metrics
        # Max elbow 165.4 (< 175.0), max shoulder 98.0 (< 100.0)
        "safe_elbow_angles": [90.5, 75.2, 110.3, 145.8, 165.4, 150.2, 130.0],
        "safe_shoulder_angles": [45.0, 60.5, 80.2, 95.5, 98.0, 85.0, 70.0],

        # Group 2: Risky stroke metrics (High injury risk)
        # Max elbow 182.5 (triggers hyperextension), max shoulder 125.0 (triggers impingement)
        "danger_elbow_angles": [90.5, 75.2, 110.3, 160.8, 182.5, 175.2, 140.0],
        "danger_shoulder_angles": [60.0, 85.0, 110.5, 125.0, 115.0, 95.0, 80.0],

        # Group 3: Abnormal data (e.g., occlusion leading to NaN)
        "nan_angles": [90.5, float('nan'), 110.3]
    }


# ==========================================
# 1. Unit Tests for SafetyRulesLayer
# ==========================================
def test_elbow_hyperextension():
    layer = SafetyRulesLayer()
    hyperextension = layer.check_elbow_hyperextension(180)
    assert hyperextension["is_safe"] is False
    assert hyperextension["max_elbow_angle"] == 180


def test_shoulder_impingement():
    layer = SafetyRulesLayer()
    impingement = layer.check_shoulder_impingement(110)
    assert impingement["is_safe"] is False
    assert impingement["max_shoulder_angle"] == 110


# ==========================================
# 2. Unit Tests for TechniqueRulesLayer
# ==========================================
def test_kinetic_chain(sample_pose_data):
    shoulder_v = sample_pose_data["shoulder_v"]
    elbow_v = sample_pose_data["elbow_v"]
    wrist_v = sample_pose_data["wrist_v"]

    layer = TechniqueRulesLayer(2.5, 2.4, 30, 5)
    result = layer.check_kinetic_chain(shoulder_v, elbow_v, wrist_v)
    assert result["is_proper"] is True
    assert result["idx_shoulder_peak"] == 3
    assert result["idx_elbow_peak"] == 4
    assert result["idx_wrist_peak"] == 5


def test_impact_point():
    layer = TechniqueRulesLayer(2.5, 2.25, 30, 5)
    result = layer.evaluate_impact_point()
    assert result["is_optimal"] is True


# ==========================================
# 3. Integration Test for DiagnosisEngine
# ==========================================
def test_diagnosis_engine_integration(sample_pose_data):
    shoulder_v = sample_pose_data["shoulder_v"]
    elbow_v = sample_pose_data["elbow_v"]
    wrist_v = sample_pose_data["wrist_v"]
    right_shoulder_angles = sample_pose_data["safe_shoulder_angles"]
    right_elbow_angles = sample_pose_data["safe_elbow_angles"]
    layer = DiagnosisEngine(2.5, 2.25, 30, 5)
    report = layer.analyze_stroke(shoulder_v, elbow_v, wrist_v, right_shoulder_angles, right_elbow_angles)
    assert report["safety_report"]["elbow_hyperextension"]["is_safe"] is True
    assert report["safety_report"]["shoulder_impingement"]["is_safe"] is True
    assert report["technique_report"]["kinetic_chain"]["is_proper"] is True
    assert report["technique_report"]["impact_point"]["is_optimal"] is True

    right_shoulder_angles = sample_pose_data["danger_shoulder_angles"]
    right_elbow_angles = sample_pose_data["danger_elbow_angles"]
    report = layer.analyze_stroke(shoulder_v, elbow_v, wrist_v, right_shoulder_angles, right_elbow_angles)
    assert report["safety_report"]["elbow_hyperextension"]["is_safe"] is False
    assert report["safety_report"]["shoulder_impingement"]["is_safe"] is False
    assert report["technique_report"]["kinetic_chain"]["is_proper"] is True
    assert report["technique_report"]["impact_point"]["is_optimal"] is True

