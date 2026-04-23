import numpy as np
from moviepy import VideoFileClip
from scipy.signal import find_peaks
import librosa
import os
from dotenv import load_dotenv
"""
intput: video file

#  Data Adapter ：
fps = 60.026
wrist_wrist_vel=
[0.03113222 0.05088606 0.05721603 0.05963984 0.06096867 0.05939249
 0.05692969 0.04784107 0.04885278 0.05403393 0.06434244 0.05712285
 0.04867457 0.03954768 0.03799048 0.03219389 0.03143368 0.02193531
 0.03138004 0.04669142 0.0656827  0.08681248 0.08672121 0.06404779
 0.04294038 0.02725376 0.02548812 0.030436   0.02889477 0.02470451
 0.01687205 0.00902402 0.00408316 0.00259265 0.00567817 0.00624747
 0.00741742 0.00722443 0.0088179  0.01723704 0.02648315 0.03699645
 0.04553605 0.05374137 0.05443137 0.0541914  0.04952771 0.04645193
 0.04524672 0.04312349 0.03702487 0.03038721 0.01748808 0.00717711
 0.00356962 0.00383223 0.00431104 0.00766732 0.01352104 0.0181104
 0.01947743 0.01809297 0.01586986 0.01513172 0.01043455 0.00214605
 0.0036994  0.01680794 0.03316031 0.04436817 0.04636232 0.04121971
 0.03530591 0.02520376 0.01633194 0.0121308  0.01197009 0.01392339
 0.01379766 0.01359646 0.01163391 0.01196161 0.01219989 0.01352973
 0.01407736 0.01480889 0.01480964 0.01582639 0.0151264  0.015281
 0.01316067 0.00944025 0.00537673 0.00362477 0.00423901 0.00554697
 0.00634112 0.00634474 0.0065486  0.0075543  0.00898692 0.01048483
 0.01062488 0.00909483 0.00773251 0.00673842 0.00679456 0.00730691
 0.00759137 0.00790035 0.00939222 0.00897658 0.0079522  0.00719889
 0.00778767 0.00812659 0.00968052 0.0103105  0.01160244 0.00974036
 0.00479871 0.0025744 ]

output: 
impact_frame
"""
load_dotenv()

class SensorFusion:
    def __init__(self, fps: float, video_path:str,audio_path:str, tolerance: int = 2):
        self._fps = fps  # Frames per second of the video, used to convert between time and frame indices
        self._TOLERANCE = tolerance  # Number of frames within which to consider an audio peak and a visual peak as matching     
        self._VIDEO_PATH = video_path
        self._AUDIO_PATH = audio_path
    def detect_impact_multimodel(self, right_wrist_vel: list[float]) -> int:
        """
        Detect the impact frame by cross-validating the detected audio peaks with the visual peaks from the wrist velocity data.
        Args:        right_wrist_vel (list[float]): A list of velocities for the right wrist joint across frames.
        Returns:        int: The index of the confirmed impact frame in the video.
        """
        video = VideoFileClip(self._VIDEO_PATH)
        # Extract audio from the video and save it as a temporary file
        video.audio.write_audiofile(self._AUDIO_PATH, logger=None)
        # Load the audio file using librosa
        y, sr = librosa.load(self._AUDIO_PATH, sr=None)
        # Now you can use the audio data (y) and sample rate (sr) for further processing
        print(f"Audio loaded successfully, sample rate: {sr}, audio shape: {y}")
        # detect the onset strength of the impact sound
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        audio_peaks_frames = librosa.util.peak_pick(
            onset_env,
            # Number of frames before and after the current frame to consider for peak picking
            pre_max=3, post_max=3,
            # Number of frames before and after the current frame to consider for calculating the average onset strength
            pre_avg=3, post_avg=5,  # before impact sound, it's quiet, after impact sound, there might be noise or echo
            delta=0.5,  # Threshold for peak picking. A higher value means that only stronger peaks will be detected.
            wait=10  # Minimum number of frames between detected peaks. Set to 0 to

        )
        audio_times_seconds = librosa.frames_to_time(audio_peaks_frames, sr=sr)
        print(self._fps)
        audio_video_frames = [int(t * self._fps) for t in
                              audio_times_seconds]  # Convert audio peak times to corresponding video frame indices
        print("--------------")
        print(audio_video_frames)

        # analyze the kinetic chain( the change in the wrist velocity) to cross-validate the detected impact sound peaks
        final_impact: int = 0
        try:
            visual_peaks, _ = find_peaks(right_wrist_vel, height=np.max(right_wrist_vel) * 0.4, distance=10
                                         # Minimum number of frames between detected peaks in the velocity data, to avoid detecting multiple peaks for a single impact event
                                         )
            print(list(visual_peaks))
            # Cross-validate the detected audio peaks with the visual peaks from the wrist velocity data

            confirmed_impacts = []
            for audio_peak in audio_video_frames:
                for visual_peak in visual_peaks:
                    if abs(audio_peak - visual_peak) <= self._TOLERANCE:
                        final_frame = int((audio_peak + visual_peak) / 2)
                        confirmed_impacts.append(final_frame)
                        break
            if confirmed_impacts:
                final_impact = confirmed_impacts[0]
                print(
                    f"Confirmed impact detected at video frame: {final_impact}, which corresponds to time: {final_impact / self._fps:.2f} seconds")

            else:
                # If no confirmed impacts are found, we can still use the visual peaks to determine the most likely impact frame based on the highest wrist velocity
                final_impact = visual_peaks[
                    int(np.argmax([right_wrist_vel[smooth_idx] for smooth_idx in visual_peaks]))]
                print(
                    f"No confirmed impacts, but the most likely impact frame based on wrist velocity is: {final_impact}, which corresponds to time: {final_impact / self._fps:.2f} seconds")

        except TypeError as e:
            print(f"Error in find_peaks: {e}")
        except Exception as e:
            print(f"Unexpected error in find_peaks: {e}")
        finally:
            if os.path.exists(self._AUDIO_PATH):
                # Remove the temporary audio file after loading it
                os.remove(self._AUDIO_PATH)
                print("Temporary audio file removed.")
            return final_impact


if __name__ == "__main__":
    from config.core import config
    fps = 60.026353033038895
    # Example usage
    sensor_fusion = SensorFusion(fps,config.get_path("VIDEO_PATH"),config.get_path("AUDIO_PATH"))
    # Simulated wrist velocity data (replace with actual data)
    right_wrist_vel = [0.03113222, 0.05088606, 0.05721603, 0.05963984, 0.06096867, 0.05939249,
                       0.05692969, 0.04784107, 0.04885278, 0.05403393, 0.06434244, 0.05712285,
                       0.04867457, 0.03954768, 0.03799048, 0.03219389, 0.03143368, 0.02193531,
                       0.03138004, 0.04669142, 0.0656827, 0.08681248, 0.08672121, 0.06404779,
                       0.04294038, 0.02725376, 0.02548812, 0.030436, 0.02889477, 0.02470451,
                       0.01687205, 0.00902402, 0.00408316, 0.00259265, 0.00567817, 0.00624747,
                       0.00741742, 0.00722443, 0.0088179, 0.01723704, 0.02648315, 0.03699645,
                       0.04553605, 0.05374137, 0.05443137, 0.0541914, 0.04952771, 0.04645193,
                       0.04524672, 0.04312349, 0.03702487, 0.03038721, 0.01748808, 0.00717711,
                       0.00356962, 0.00383223, 0.00431104, 0.00766732, 0.01352104, 0.0181104,
                       0.01947743, 0.01809297, 0.01586986, 0.01513172, 0.01043455, 0.00214605,
                       0.0036994, 0.01680794, 0.03316031, 0.04436817, 0.04636232, 0.04121971,
                       0.03530591, 0.02520376, 0.01633194, 0.0121308, 0.01197009, 0.01392339,
                       0.01379766, 0.01359646, 0.01163391, 0.01196161, 0.01219989, 0.01352973,
                       0.01407736, 0.01480889, 0.01480964, 0.01582639, 0.0151264, 0.015281,
                       0.01316067, 0.00944025, 0.00537673, 0.00362477, 0.00423901, 0.00554697,
                       0.00634112, 0.00634474, 0.0065486, 0.0075543, 0.00898692, 0.01048483,
                       0.01062488, 0.00909483, 0.00773251, 0.00673842, 0.00679456, 0.00730691,
                       0.00759137, 0.00790035, 0.00939222, 0.00897658, 0.0079522, 0.00719889,
                       0.00778767, 0.00812659, 0.00968052, 0.0103105, 0.01160244, 0.00974036,
                       0.00479871, 0.0025744]
    impact_frame = sensor_fusion.detect_impact_multimodel(right_wrist_vel=right_wrist_vel)
    print(f"Detected impact frame: {impact_frame}")
