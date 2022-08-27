import cv2
import mediapipe as mp
import numpy as np
import numpy.linalg as linalg
from numpy.linalg import norm
import matplotlib.pyplot as plt
from math import cos, sin, acos, asin, sqrt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import random
from PIL import Image

parts = ["nose", "left eye", "right eye", "left ear", "right ear", "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"]
part_to_ind = {part: i for i, part in enumerate(parts)}
frames = []

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

# angle of a vector
def angle(v1: np.array, v2: np.array) -> float:
    return acos(np.dot(v1, v2) / norm(v1) / norm(v2))

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

def get_stats(vecs: np.array):
    # angle of the left elbow
    lef_elbow_angle = angle(vecs[11] - vecs[13], vecs[15] - vecs[13])
    # angle of the right elbow
    rig_elbow_angle = angle(vecs[12] - vecs[14], vecs[16] - vecs[14])
    torso_angle = (angle(vecs[25] - vecs[23], vecs[11] - vecs[23]) + angle(vecs[26] - vecs[24], vecs[12] - vecs[24])) / 2

    # torso dimensions
    torso = (vecs[24]-vecs[12], vecs[23]-vecs[11], vecs[12]-vecs[11], vecs[24]-vecs[23]) # right, left, top, bottom

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

def collect_frames(name: str, ratio: float):
    cap = cv2.VideoCapture(name)
    frames = []
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)

    return frames[:int(len(frames) * ratio)]

no_pushup = collect_frames("no-pushup-0.mp4", 1) + collect_frames("no-pushup-1.mp4", 1)
pushup = collect_frames("pushup-0.mp4", 0.66) + collect_frames("pushup-1.mp4", 0.66)

no_pushup_meta = []
pushup_meta = []

for i in no_pushup:
    cur_pose = get_pose_info(i)
    if cur_pose:
        no_pushup_meta.append(get_stats(cur_pose[0]))

for i in pushup:
    cur_pose = get_pose_info(i)
    if cur_pose:
        pushup_meta.append(get_stats(cur_pose[0]))

no_pushup_vector = np.array([[v for k, v in i.items()] for i in no_pushup_meta])
pushup_vector = np.array([[v for k, v in i.items()] for i in pushup_meta])

data = []
for i in no_pushup_vector:
    if not np.isnan(i).any():
        data.append((i, 0))
for i in pushup_vector:
    if not np.isnan(i).any():
        data.append((i, 1))

random.shuffle(data)
num_train = int(0.66 * len(data))
train_data, test_data = data[:num_train], data[num_train:]

x_train, y_train = zip(*train_data)
x_test, y_test = zip(*test_data)

x_train = np.array(x_train).astype("float32")
y_train = np.array(y_train).astype("int32")
x_test = np.array(x_test).astype("float32")
y_test = np.array(y_test).astype("int32")

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

inputs = keras.Input((6,))
x = layers.BatchNormalization()(inputs)
x = layers.Dense(10, activation="relu", kernel_regularizer="l1_l2")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(x=np.array(x_train), y=y_train, validation_data=(np.array(x_test), y_test), epochs=100, batch_size=32, callbacks=[keras.callbacks.ModelCheckpoint(filepath="model.keras", save_best_only=True)])

other_model = keras.models.load_model("model.keras")
print(other_model.evaluate(x=x_test, y=y_test))