import mediapipe as mp
import math

def calculateAngle(landmark1, landmark2, landmark3):
    if len(landmark1) == 2:
        x1, y1 = landmark1
        z1 = 0
    else:
        x1, y1, z1 = landmark1

    if len(landmark2) == 2:
        x2, y2 = landmark2
        z2 = 0
    else:
        x2, y2, z2 = landmark2

    if len(landmark3) == 2:
        x3, y3 = landmark3
        z3 = 0
    else:
        x3, y3, z3 = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

def calculateAngle2(landmark1, landmark2, landmark3):
    if len(landmark1) == 2:
        x1, y1 = landmark1
        z1 = 0
    else:
        x1, y1, z1 = landmark1

    if len(landmark2) == 2:
        x2, y2 = landmark2
        z2 = 0
    else:
        x2, y2, z2 = landmark2

    if len(landmark3) == 2:
        x3, y3 = landmark3
        z3 = 0
    else:
        x3, y3, z3 = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    angle = 360 - angle
    if angle >= 360:
        angle -= 360
    return angle

def calculateDistance(landmark1, landmark2):
    if len(landmark1) == 2:
        x1, y1 = landmark1
        z1 = 0
    else:
        x1, y1, z1 = landmark1

    if len(landmark2) == 2:
        x2, y2 = landmark2
        z2 = 0
    else:
        x2, y2, z2 = landmark2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_pose_landmark_points():
    mp_pose = mp.solutions.pose
    return [                   
            mp_pose.PoseLandmark.NOSE.value,#0
            mp_pose.PoseLandmark.LEFT_EYE_INNER.value,#1
            mp_pose.PoseLandmark.LEFT_EYE.value,#2
            mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,#3
            mp_pose.PoseLandmark.RIGHT_EYE_INNER.value,#4
            mp_pose.PoseLandmark.RIGHT_EYE.value,#5
            mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,#6
            mp_pose.PoseLandmark.LEFT_EAR.value,#7
            mp_pose.PoseLandmark.RIGHT_EAR.value,#8
            mp_pose.PoseLandmark.MOUTH_LEFT.value,#9
            mp_pose.PoseLandmark.MOUTH_RIGHT.value,#10
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,#11
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,#12
            mp_pose.PoseLandmark.LEFT_ELBOW.value,#13
            mp_pose.PoseLandmark.RIGHT_ELBOW.value,#14
            mp_pose.PoseLandmark.LEFT_WRIST.value,#15
            mp_pose.PoseLandmark.RIGHT_WRIST.value,#16
            mp_pose.PoseLandmark.LEFT_PINKY.value,#17
            mp_pose.PoseLandmark.RIGHT_PINKY.value,#18
            mp_pose.PoseLandmark.LEFT_INDEX.value,#19
            mp_pose.PoseLandmark.RIGHT_INDEX.value,#20
            mp_pose.PoseLandmark.LEFT_THUMB.value,#21
            mp_pose.PoseLandmark.RIGHT_THUMB.value,#22
            mp_pose.PoseLandmark.LEFT_HIP.value,#23
            mp_pose.PoseLandmark.RIGHT_HIP.value,#24
            mp_pose.PoseLandmark.LEFT_KNEE.value,#25
            mp_pose.PoseLandmark.RIGHT_KNEE.value,#26
            mp_pose.PoseLandmark.LEFT_ANKLE.value,#27
            mp_pose.PoseLandmark.RIGHT_ANKLE.value,#28
            mp_pose.PoseLandmark.LEFT_HEEL.value,#29
            mp_pose.PoseLandmark.RIGHT_HEEL.value,#30
            mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,#31
            mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value#32
        ]