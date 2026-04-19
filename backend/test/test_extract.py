import pytest
from utils.extract import *


@pytest.fixture
def sample_pose_data():
    """
    Simulated 3D pose data for a badminton stroke.
    Coordinates are [x, y, z]. In normalized coordinates, lower 'y' means higher physical position.
    """
    return {
        # Frame 0: Preparation (arm bent, wrist below shoulder level)
        0: {
            'joints':
            {
                "right_shoulder": [0.50, 0.50, 0.50],
             "right_elbow": [0.60, 0.60, 0.40],
             "right_wrist": [0.65, 0.70, 0.45]
             }
        },
        # Frame 1: Raising Arm (backswing with elbow raised, wrist lagging)
        1: {
            'joints':
            {
                "right_shoulder": [0.50, 0.48, 0.50],
             "right_elbow": [0.65, 0.30, 0.45],
             "right_wrist": [0.60, 0.40, 0.60]
             }
        },
        # Frame 2: Pre-Impact (加速挥拍，手臂开始向上伸展)
        2: {
            'joints':
            {
                "right_shoulder": [0.50, 0.45, 0.50],
             "right_elbow": [0.60, 0.20, 0.55],
             "right_wrist": [0.70, 0.15, 0.60]
            }
        },
        # Frame 3: Impact Frame (moment of impact, wrist at highest point with min y-value, arm nearly fully extended)
        3: {
            'joints':
            {
                "right_shoulder": [0.50, 0.45, 0.50],
             "right_elbow": [0.60, 0.25, 0.55],
             "right_wrist": [0.70, 0.05, 0.60]
            }
        },
        # Frame 4: Follow-through (arm drops and crosses in front of the body)
        4: {
            'joints':
            {
            "right_shoulder": [0.50, 0.46, 0.50],
             "right_elbow": [0.40, 0.50, 0.45],
             "right_wrist": [0.30, 0.65, 0.40]
            }
        }
    }


def test_extract_impact_metrics(sample_pose_data):
    impact_metrics = ImpactMetrics(2)
    right_shoulder = extract_body_parts(sample_pose_data, "right_shoulder")
    right_elbow = extract_body_parts(sample_pose_data, "right_elbow")
    right_wrist = extract_body_parts(sample_pose_data, "right_wrist")
    threshold = impact_metrics.extract_impact_threshold(right_shoulder, right_elbow, right_wrist)
    arm_extension_length=impact_metrics.extract_arm_extension_length(right_shoulder, right_wrist)

    # 1. Assert Types (Sanity Check)
    assert isinstance(threshold, float)
    assert isinstance(arm_extension_length, float)

    # 2. Assert Physical Logic (Triangle Inequality Theorem)
    # The straight-line distance (shoulder to wrist) can NEVER be longer than the sum of the two segments (shoulder to elbow + elbow to wrist).
    assert arm_extension_length <= threshold

    # 3. Assert Exact Values (Based on the math for Frame 2 in your sample_pose_data)
    # Frame 2: Shoulder=[0.50, 0.45, 0.50], Elbow=[0.60, 0.20, 0.55], Wrist=[0.70, 0.15, 0.60]
    # Calculated and rounded to 3 decimal places
    assert threshold == 0.396
    assert arm_extension_length == 0.374
