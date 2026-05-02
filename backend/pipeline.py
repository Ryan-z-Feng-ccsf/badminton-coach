from src.cv.pose_detector import DetectorEngine
from src.analysis.rules_engine import DiagnosisEngine
from src.analysis.sensor_fusion import SensorFusion
from utils import extract
from utils.extract import VelocityMetrics, Metrics
from config.core import config



def format_report(video_path:str) -> dict:
    engine = DetectorEngine()
    metadata = engine.get_metadata(config.get_path("POSE_MODEL_PATH"),video_path)
    # Extract fps
    fps = metadata['fps']
    # Extract pose data payload
    pose_data_payload = extract.extract_pose_data_payload(metadata)
    vel_metric = VelocityMetrics()
    visual_impact = vel_metric.extract_joint_velocity(pose_data_payload, 'right_wrist', fps)
    sensor_fusion = SensorFusion(fps,video_path,config.get_path("AUDIO_PATH"))
    # Extract impact frame
    impact_frame = sensor_fusion.detect_impact_multimodel(visual_impact)
    # Extract metrics
    metrics = Metrics(impact_frame)
    # TODO:
    # add the right knee angle and right ankle angle
    extract_metrics = metrics.extract_metrics(extract.extract_body_parts(pose_data_payload, 'right_shoulder'),
                                              extract.extract_body_parts(pose_data_payload, 'right_elbow'),
                                              extract.extract_body_parts(pose_data_payload, 'right_wrist'),
                                              extract.extract_body_parts(pose_data_payload, 'right_hip'),
                                              extract.extract_body_parts(pose_data_payload, 'right_knee'),
                                              extract.extract_body_parts(pose_data_payload, 'right_ankle'),
                                              pose_data_payload,
                                              fps
                                              )
    # Diagnose
    diagnosis = DiagnosisEngine(
        extract_metrics['impact_threshold'],
        extract_metrics['arm_extension_length'],
        fps,
        impact_frame
    )
    report = diagnosis.analyze_stroke(extract_metrics)
    return report

if __name__ == "__main__":
    print(format_report(config.get_path("VIDEO_PATH")))
