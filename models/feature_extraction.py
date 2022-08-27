import cv2
import pickle
import mediapipe as mp
import numpy as np
import numpy.linalg as linalg
from numpy.linalg import norm
import matplotlib.pyplot as plt
from math import cos, sin, acos, asin, sqrt
from preprocessing import *

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

file = "data/no-pushup-0.mp4"

frames = collect_frames(file, 1)
features = [Pushup.get_stats(get_pose_info(i)[0]) for i in frames if get_pose_info(i) and not get_pose_info(i) is None]

with open(f"{file}.pkl", "wb") as f:
    pickle.dump(features, f)

# the parameters of the pose
pose_params = {}
# the frames that we use
used_frames = []

for image in frames:
    cur_pose = get_pose_info(image)
    if cur_pose is not None:
        # add the pose and its pose information
        used_frames.append(image)
        vecs, landmarks = cur_pose
        for name, value in Pushup.get_stats(vecs).items():
            if name not in pose_params:
                pose_params[name] = []
            pose_params[name].append(value)

# quantisize it
for name in pose_params:
    pose_params[name] = np.array(pose_params[name])

for i in range(0, len(used_frames), 5):
    # plot the image along with the pose parameters
    image = used_frames[i]
    fig = plt.figure(figsize=(10, 5))

    fig.add_subplot(121)
    plt.imshow(image)

    fig.add_subplot(122)
    plt.title("Pose Parameters")

    ci = 0
    ways = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for name, params in pose_params.items():
        plt.plot(range(len(params)), params / params.max(), ways[ci % len(ways)], label=name)
        ci += 1
    plt.legend()
    plt.axvline(x=i)
    plt.show()
