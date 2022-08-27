import numpy as np
from math import acos, sqrt
import mediapipe as mp
import numpy.linalg as linalg
from numpy.linalg import norm

# angle of a vector
def angle(v1: np.array, v2: np.array) -> float:
    return acos(np.clip(np.dot(v1, v2) / norm(v1) / norm(v2), -1, 1))

def dist(v1: np.array, v2: np.array, v3: np.array = None) -> float:
    """
        If v3 is not defined, then this is equal to the distance from the line Origin-V1 to the point V2\n
        If v3 is defined, then this is equal to the distance from the line v1-v2 to v3
    """
    if v3 is not None:
        return dist(v2-v1, v3-v1)
    else:
        # cos = v1*v2/|v1|/|v2|; sin = sqrt(1-cos^2); dist=sin |v2|
        return sqrt(1 - (np.dot(v1, v2) / norm(v1) / norm(v2)) ** 2) * norm(v2)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_pose_info(image):
    results = pose.process(image)
    p_landmarks = results.pose_landmarks
    if not p_landmarks:
        return None
    x_pos = []
    y_pos = []

    for i in str(p_landmarks).split('landmark')[1:]:
        i_1 = i.split()
        x_pos.append(int(640 * float(i_1[2])))
        y_pos.append(int(480 * float(i_1[4])))

    return np.array(list(zip(x_pos, y_pos))), p_landmarks

class Preprocessing:
    @staticmethod
    def get_stats(vecs: np.array) -> dict:
        raise NotImplementedError("A subclass should implement this")

    @staticmethod
    def preprocess(vecs: np.array) -> np.array:
        raise NotImplementedError("A subclass should implement this")

class Pushup(Preprocessing):
    @staticmethod
    def get_stats(vecs: np.array) -> dict:
        # angle of the left elbow
        lef_elbow_angle = angle(vecs[11] - vecs[13], vecs[15] - vecs[13])
        # angle of the right elbow
        rig_elbow_angle = angle(vecs[12] - vecs[14], vecs[16] - vecs[14])
        torso_angle = (angle(vecs[25] - vecs[23], vecs[11] - vecs[23]) + angle(vecs[26] - vecs[24],
                                                                               vecs[12] - vecs[24])) / 2

        # torso dimensions
        torso = (vecs[24] - vecs[12], vecs[23] - vecs[11], vecs[12] - vecs[11], vecs[24] - vecs[23])  # right, left, top, bottom

        # torso height / torso width; useful for detecting if posture is sideways
        torso_ratio = (linalg.norm(torso[0]) + linalg.norm(torso[1])) / (linalg.norm(torso[2]) + linalg.norm(torso[3]))
        # average sine of the torso - measures the "angle" of the body
        torso_sin = (abs(torso[0][1] / linalg.norm(torso[0])) + abs(torso[1][1] / linalg.norm(torso[1]))) / 2

        # distance of the chest from the line formed by the fingers and toes
        chest_dis = linalg.norm(np.cross(vecs[20] - vecs[32], vecs[32] - vecs[12])) / linalg.norm(vecs[20] - vecs[32])

        return {
            "lef_ea": lef_elbow_angle,
            "rig_ea": rig_elbow_angle,
            "ta": torso_angle,
            "tr": torso_ratio,
            "ts": torso_sin,
            "cdis": chest_dis
        }

    @staticmethod
    def preprocess(vecs: np.array) -> np.array:
        stats = Pushup.get_stats(vecs)
        names = ['lef_ea', 'rig_ea', 'ta', 'tr', 'ts', 'cdis']
        return np.array([stats[i] for i in names])

class Situp(Preprocessing):
    @staticmethod
    def get_stats(vecs: np.array) -> dict:
        rig_hip_angle = angle(vecs[12] - vecs[24], vecs[26] - vecs[24])
        lef_hip_angle = angle(vecs[25] - vecs[23], vecs[11] - vecs[23])

        rig_knee_angle = angle(vecs[28] - vecs[26], vecs[24] - vecs[26])
        lef_knee_angle = angle(vecs[27] - vecs[25], vecs[23] - vecs[25])
        avg_knee_angle = (rig_knee_angle + lef_knee_angle) / 2

        torso = (vecs[24] - vecs[12], vecs[23] - vecs[11], vecs[12] - vecs[11], vecs[24] - vecs[23])  # right, left, top, bottom
        thighs = (vecs[26] - vecs[24], vecs[25] - vecs[23]) # right, left
        knees = (vecs[28] - vecs[26], vecs[27] - vecs[25]) # right, left

        # torso height / torso width; useful for detecting if posture is sideways
        torso_ratio = (linalg.norm(torso[0]) + linalg.norm(torso[1])) / (linalg.norm(torso[2]) + linalg.norm(torso[3]))

        # sine of various body parts
        rig_thigh_sin = thighs[0][1] / norm(thighs[0])
        lef_thigh_sin = thighs[1][1] / norm(thighs[1])

        rig_knee_sin = knees[0][1] / norm(knees[0])
        lef_knee_sin = knees[1][1] / norm(knees[1])
        avg_knee_sin = (rig_knee_sin + lef_knee_sin) / 2

        return {
            "r_ha": rig_hip_angle,
            "l_ha": lef_hip_angle,
            "ka": avg_knee_angle,
            "tr": torso_ratio,
            "r_ts": rig_thigh_sin,
            "l_ts": lef_thigh_sin,
            "ks": avg_knee_sin
        }

    @staticmethod
    def preprocess(vecs: np.array) -> np.array:
        stats = Situp.get_stats(vecs)
        names = ["r_ha", "l_ha", "ka", "tr", "r_ts", "l_ts", "ks"]
        return np.array([stats[i] for i in names])

class Squat(Preprocessing):
    @staticmethod
    def get_stats(vecs: np.array) -> dict:
        thighs = (vecs[26] - vecs[24], vecs[25] - vecs[23]) # right, left
        torso = (vecs[24] - vecs[12], vecs[23] - vecs[11], vecs[12] - vecs[11], vecs[24] - vecs[23])  # right, left, top, bottom

        # torso height / torso width; useful for detecting if posture is sideways
        torso_ratio = (linalg.norm(torso[0]) + linalg.norm(torso[1])) / (linalg.norm(torso[2]) + linalg.norm(torso[3]))
        torso_sin = (abs(torso[0][1] / linalg.norm(torso[0])) + abs(torso[1][1] / linalg.norm(torso[1]))) / 2

        # sine of various body parts
        rig_thigh_sin = thighs[0][1] / norm(thighs[0])
        lef_thigh_sin = thighs[1][1] / norm(thighs[1])

        rig_knee_angle = angle(vecs[28] - vecs[26], vecs[24] - vecs[26])
        lef_knee_angle = angle(vecs[27] - vecs[25], vecs[23] - vecs[25])

        rig_hip_angle = angle(vecs[12] - vecs[24], vecs[26] - vecs[24])
        lef_hip_angle = angle(vecs[25] - vecs[23], vecs[11] - vecs[23])
        avg_hip_angle = (lef_hip_angle + rig_hip_angle) / 2
        return {
            "tr": torso_ratio,
            "ts": torso_sin,
            "r_ts": rig_thigh_sin,
            "l_ts": lef_thigh_sin,
            "r_ka": rig_knee_angle,
            "l_ka": lef_knee_angle,
            "ha": avg_hip_angle
        }

    @staticmethod
    def preprocess(vecs: np.array) -> np.array:
        names = ["tr", "ts", "r_ts", "l_ts", "r_ka", "l_ka", "ha"]
        stats = Squat.get_stats(vecs)
        return np.array([stats[i] for i in names])

class BicepCurl(Preprocessing):
    @staticmethod
    def get_stats(vecs: np.array) -> dict:
        torso = (vecs[24] - vecs[12], vecs[23] - vecs[11], vecs[12] - vecs[11], vecs[24] - vecs[23])  # right, left, top, bottom

        torso_ratio = (linalg.norm(torso[0]) + linalg.norm(torso[1])) / (linalg.norm(torso[2]) + linalg.norm(torso[3]))

        rig_hip_angle = angle(vecs[12] - vecs[24], vecs[26] - vecs[24])
        lef_hip_angle = angle(vecs[25] - vecs[23], vecs[11] - vecs[23])
        avg_hip_angle = (lef_hip_angle + rig_hip_angle) / 2

        thighs = (vecs[26] - vecs[24], vecs[25] - vecs[23])  # right, left
        rig_thigh_sin = thighs[0][1] / norm(thighs[0])
        lef_thigh_sin = thighs[1][1] / norm(thighs[1])
        avg_thigh_sin = (rig_thigh_sin + lef_thigh_sin) / 2

        lef_elbow_angle = angle(vecs[11] - vecs[13], vecs[15] - vecs[13])
        rig_elbow_angle = angle(vecs[12] - vecs[14], vecs[16] - vecs[14])

        rig_shoulder_angle = angle(vecs[14] - vecs[12], vecs[24] - vecs[12])
        lef_shoulder_angle = angle(vecs[13] - vecs[11], vecs[23] - vecs[11])
        avg_shoulder_angle = (lef_shoulder_angle + rig_shoulder_angle) / 2

        return {
            "tr": torso_ratio,
            "ha": avg_hip_angle,
            "th-s": avg_thigh_sin,
            "l-ea": lef_elbow_angle,
            "r-ea": rig_elbow_angle,
            "sa": avg_shoulder_angle,
        }

    @staticmethod
    def preprocess(vecs: np.array) -> np.array:
        names = ["tr", "ha", "th-s", "l-ea", "r-ea", "sa"]
        stats = BicepCurl.get_stats(vecs)
        return np.array([stats[i] for i in names])