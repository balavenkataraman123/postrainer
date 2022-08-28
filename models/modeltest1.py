from cmath import e
from os import stat
import cv2
import mediapipe as mp
import numpy as np
import numpy.linalg as linalg
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time
from math import cos, sin, acos, asin, sqrt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import random

from typing import List

from preprocessing import *
from PIL import Image

motivquotes = [
    " \"The last three or four reps is what makes the muscle grow. This area of pain divides a champion from someone who is not a champion \"-Arnold Schwarzenegger",
    " \"The successful warrior is the average man, with laser-like focus.\" -Bruce Lee",
    " \"The pain you feel today will be the strength you feel tomorrow.\" -Arnold Schwarzenegger",
    " \"You miss 100% of the shots you don\'t take.\" -Wayne Gretzky",
    "\"Blood, sweat and respect. First two you give. Last one you earn.\" -The Rock"
]


class Screen:
    def __init__(self):
        self.increment = 40

    @staticmethod
    def query_model(model: keras.Model, vecs: np.array, workout: Preprocessing):
        return model(workout.preprocess(vecs).reshape((1, -1))).numpy()[0, 0]

    def augment_with_message(self, image: np.array, messages: List[str]):
        for message, y in zip(messages, range(120, 120 + 40 * len(messages), 40)):
            image = cv2.putText(image, message, (20, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return image[:,:,:3].astype("uint8")

    @staticmethod
    def draw_border(image: np.array, color: list) -> np.array:
        return cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=color)


class SitupScreen(Screen):
    def __init__(self):
        super().__init__()
        self.model = keras.models.load_model("situp-model.keras")
        self.last_state = None
        self.state_torso = []
        self.tot_situp = 0
        self.last_up = 0

        self.high_enough = True
        self.low_enough = True

        self.up_boundary = 1.35
        self.down_boundary = 1.8
        self.low_requirement = 0.1
        self.high_requirement = 0.95
        self.down_time = 5

    def render(self, image) -> np.array:
        global minvis, reps, exercise, jctimer, thismotquote, justchanged
        if justchanged == 1:
            jctimer = 400
            thismotquote = random.choice(motivquotes)
            justchanged = 0
        if jctimer == 0:

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            parts = None
            minvis = 1
            try:
                a = get_pose_info(image)
                parts = a[0]
                minvis = a[1]
            except:
                minvis = 0.69

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_AREA)

            if parts is not None:
                (vecs, landmarks) = parts
                mp_drawing.draw_landmarks(
                    image,
                    landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                stats = Situp.get_stats(vecs)
                if minvis <= 0.2:
                    image = cv2.putText(image, "The camera cannot see you well", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                
                in_position = Screen.query_model(self.model, vecs, Situp())
                # have we detected the person
                if in_position >= 0.5:
                    print(stats)
                    # get some measurements
                    torso_angle = (stats['r_ha'] + stats['l_ha']) / 2
                    torso_sin = stats['ts']

                    if torso_angle <= self.up_boundary:
                        # now in "up state"
                        if self.last_state is "down":
                            # with all the data from the "down-time", we check to see if the person went low enough
                            self.low_enough = False
                            for i in self.state_torso:
                                if i <= self.down_boundary:
                                    self.low_enough = True
                            self.state_torso.clear()
                            self.tot_situp += 1

                            if self.tot_situp == reps[exercise]:
                                justchanged = 1
                                exercise += 1
                        self.last_up = time.time()
                        self.last_state = "up"

                        self.state_torso.append(torso_sin)
                    if torso_angle >= self.down_boundary:
                        # now in "down state"
                        if self.last_state is "up":
                            # check if the person went high enough when they went up
                            self.high_enough = False
                            for i in self.state_torso:
                                if i >= self.high_requirement:
                                    self.high_enough = True
                            self.state_torso.clear()
                        self.last_state = "down"

                        self.state_torso.append(torso_sin)

                    messages = []
                    if not self.low_enough:
                        messages.append("When going down, make sure to be flat to the ground")
                    if not self.high_enough:
                        messages.append("When going up, make sure to come all the way up")
                    if time.time() - self.last_up >= self.down_time:
                        messages.append("Let's do another sit up (if you're already doing them, come higher up)!!")

                    image = self.augment_with_message(image, messages)
                    image = Screen.draw_border(image, [0, 255, 0])

                    scores.append(in_position)
                    image = cv2.putText(image, "YOUR SCORE: " + str(in_position), (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    image = Screen.draw_border(image, [0, 0, 255])

                image = cv2.putText(image, str(self.tot_situp), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
        else:
            jctimer -= 1
            image = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_AREA)
            image = cv2.putText(image, "DO " + str(reps[exercise]) + " SIT UPS", (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            image = cv2.putText(image, "put your laptop on the floor, about 6 feet away from you", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, )
            image = cv2.putText(image, "do your sit up parallel to your screen", (10, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, )

            image = cv2.putText(image, thismotquote, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                                cv2.LINE_AA, )
        return image


class SquatScreen(Screen):
    def __init__(self):
        super().__init__()
        self.model = keras.models.load_model("squat-model.keras")
        self.last_state = None
        self.state_angle = []
        self.tot_squat = 0
        self.last_down = 0

        self.high_enough = True
        self.low_enough = True
        self.back_straight = True

        self.up_boundary = 2.35
        self.down_boundary = 1.5
        self.low_requirement = 1.2
        self.high_requirement = 2.7
        self.down_time = 5

    def render(self, image) -> np.array:
        global minvis, reps, exercise, jctimer, thismotquote, justchanged
        if justchanged == 1:
            jctimer = 400
            thismotquote = random.choice(motivquotes)
            justchanged = 0
        if jctimer == 0:

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            parts = None
            minvis = 1
            try:
                a = get_pose_info(image)
                parts = a[0]
                minvis = a[1]
            except:
                minvis = 0.69

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_AREA)

            if parts is not None:
                (vecs, landmarks) = parts
                mp_drawing.draw_landmarks(
                    image,
                    landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                stats = Squat.get_stats(vecs)
                if minvis <= 0.2:
                    image = cv2.putText(image, "The camera cannot see you well", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)

                in_position = Screen.query_model(self.model, vecs, Squat())
                # have we detected the person
                if in_position >= 0.5:
                    # get some measurements
                    knee_angle = (stats['r_ka'] + stats['l_ka']) / 2

                    torso_bot, torso_top = (vecs[23] + vecs[24]) / 2, (vecs[12] + vecs[11]) / 2
                    avg_torso_vec = torso_top - torso_bot
                    torso_sin = abs(avg_torso_vec[1] / norm(avg_torso_vec))
                    print(knee_angle, torso_sin)
                    self.back_straight = torso_sin >= 0.7

                    if knee_angle >= self.up_boundary:
                        # now in "up state"
                        if self.last_state is "down":
                            # with all the data from the "down-time", we check to see if the person went low enough
                            self.low_enough = False
                            for i in self.state_angle:
                                if i <= self.low_requirement:
                                    self.low_enough = True
                            self.state_angle.clear()
                            self.tot_squat += 1

                            if self.tot_squat == reps[exercise]:
                                justchanged = 1
                                exercise += 1
                        self.last_down = time.time()
                        self.last_state = "up"

                        self.state_angle.append(knee_angle)
                    if knee_angle <= self.down_boundary:
                        # now in "down state"
                        if self.last_state is "up":
                            # check if the person went high enough when they went up
                            self.high_enough = False
                            for i in self.state_angle:
                                if i >= self.high_requirement:
                                    self.high_enough = True
                            self.state_angle.clear()
                        self.last_state = "down"

                        self.state_angle.append(knee_angle)

                    messages = []
                    if not self.low_enough:
                        messages.append("When going down, make sure to make a 90 degree angle")
                    if not self.high_enough:
                        messages.append("When going up, make sure to come all the way up")
                    if not self.back_straight:
                        messages.append("Make sure your back stays straight (more vertical)")
                    if time.time() - self.last_down >= self.down_time:
                        messages.append("Let's do another squat (or increase your movement)!!")

                    image = self.augment_with_message(image, messages)
                    image = Screen.draw_border(image, [0, 255, 0])

                    scores.append(in_position)
                    image = cv2.putText(image, "YOUR SCORE: " + str(in_position), (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    image = Screen.draw_border(image, [0, 0, 255])

                image = cv2.putText(image, str(self.tot_squat), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                    cv2.LINE_AA)
        else:
            jctimer -= 1
            image = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_AREA)
            image = cv2.putText(image, "DO " + str(reps[exercise]) + " SQUATS", (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            image = cv2.putText(image, "put your laptop on a table 6 feet away from you", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, )
            image = cv2.putText(image, "do your squat facing sideways", (10, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, )

            image = cv2.putText(image, thismotquote, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                                cv2.LINE_AA, )
        return image

class PushupScreen(Screen):
    def __init__(self):
        super().__init__()
        self.numpushup = 0
        self.currstate = 1  # 1 for up and 0 for down
        self.past40elbow = []

    def render(self, image: np.array) -> np.array:
        global exercise, justchanged, reps, thismotquote, jctimer, minvis
        if justchanged == 1:
            jctimer = 400
            thismotquote = random.choice(motivquotes)
            justchanged = 0
        if jctimer == 0:
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            parts = None
            minvis = 1
            try:
                a = get_pose_info(image)
                parts = a[0]
                minvis = a[1]
            except:
                minvis = 0.69

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_AREA)
            landmarks = None
            if parts is not None:
                (vecs, landmarks) = parts
                stats = (Pushup.get_stats(vecs))
                elbow_angle = (stats["cdis"])
                trsa = stats["ta"]
                image.flags.writeable = True
                input_vector = Pushup.preprocess(vecs).reshape((1, -1))

                prediction = model(input_vector).numpy()[0, 0]
                image = cv2.putText(image, str(self.numpushup), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                    cv2.LINE_AA)

                if minvis <= 0.2:
                    image = cv2.putText(image, "The camera cannot see you well", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                if prediction > 0.5:
                    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0, 255, 0])
                    print(elbow_angle)
                    self.past40elbow.append(elbow_angle)

                    if len(self.past40elbow) > 40:
                        self.past40elbow.pop(0)

                    if elbow_angle <= 110 and self.currstate == 1:
                        self.currstate = 0
                    if elbow_angle >= 190 and self.currstate == 0:
                        self.currstate = 1
                        self.numpushup += 1
                    if self.numpushup == reps[exercise]:
                        justchanged = 1
                        exercise += 1
                    if min(self.past40elbow) > 110:
                        image = cv2.putText(image, "go down lower ", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2, cv2.LINE_AA)
                    if len(self.past40elbow) > 20 and trsa < 2.5:
                        image = cv2.putText(image, "make sure your back is straight", (20, 240),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    scores.append(prediction)
                    image = cv2.putText(image, "YOUR SCORE: " + str(prediction), (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 255])

            mp_drawing.draw_landmarks(
                image,
                landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        else:
            jctimer -= 1
            image = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_AREA)
            image = cv2.putText(image, "DO " + str(reps[exercise]) + " " + exercises[exercise], (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if exercises[exercise] == "PUSHUPS":
                image = cv2.putText(image, "put your laptop on the floor, about 6 feet away from you", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, )
                image = cv2.putText(image, "do your pushup parallel to your screen", (10, 115),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, )

            image = cv2.putText(image, thismotquote, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                                cv2.LINE_AA, )
        return image

model = keras.models.load_model("pushup-model.keras")

# define the camera
cap = cv2.VideoCapture(0)

exercise = 0
justchanged = 1
minvis = 1
exercises = ["SITUPS", "PUSHUPS", "SQUATS"]
reps = [10, 10, 10]
jctimer = 0
thismotquote = ""
scores = []

situp_screen = SitupScreen()
pushup_screen = PushupScreen()
squat_screen = SquatScreen()
while cap.isOpened():
    if exercise >= len(exercises):
        with open("scores.txt", "wt") as f:
            avg_score = sum(scores) / len(scores)
            f.write(str(avg_score))
        break

    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    if exercises[exercise] == "PUSHUPS":
        image = pushup_screen.render(image)
    elif exercises[exercise] == "SITUPS":
        image = situp_screen.render(image)
    elif exercises[exercise] == "SQUATS":
        image = squat_screen.render(image)
    cv2.imshow("Image", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
