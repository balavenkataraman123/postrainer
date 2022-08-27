# Just adding a few features here - I'm hoping for a common sense baseline to be made here

import cv2
import mediapipe as mp
import numpy as np
import numpy.linalg as linalg
from numpy.linalg import norm
import matplotlib.pyplot as plt
from math import cos, sin, acos, asin, sqrt
from preprocessing import *

parts = ["nose", "left eye", "right eye", "left ear", "right ear", "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"]
part_to_ind = {part: i for i, part in enumerate(parts)}
frames = []

# define the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frames.append(image)
    parts = get_pose_info(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if parts is not None:
        vecs, landmarks = parts
        print(BicepCurl.get_stats(vecs))

        image.flags.writeable = True
        mp_drawing.draw_landmarks(
            image,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    cv2.imshow("Image", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

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
        for name, value in BicepCurl.get_stats(vecs).items():
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
    ways = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
    for name, params in pose_params.items():
        plt.plot(range(len(params)), params / params.max(), ways[ci % len(ways)], label=name)
        ci += 1
    plt.legend()
    plt.axvline(x=i)
    plt.show()
