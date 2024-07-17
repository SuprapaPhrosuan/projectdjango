from django.shortcuts import render
import cv2
import math
import mediapipe as mp
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
from django.views.decorators.gzip import gzip_page
import pygame
import os
from django.conf import settings 
from time import time
import matplotlib.pyplot as plt

start_time = None
total_time = 0 
is_doing = False
import mediapipe.python.solutions.drawing_utils as test2
mp_drawing = test2
import mediapipe.python.solutions.pose as test
mp_pose = test

pygame.mixer.init()
sound_file_path0 = os.path.join(settings.STATIC_ROOT, 'simple-notification-152054.mp3')
sound_file_path1 = os.path.join(settings.STATIC_ROOT, 'Correct Answer Sound Effect.mp3')
sound_file_path2 = os.path.join(settings.STATIC_ROOT, 'Alerts.wav')
sound_file_path3 = os.path.join(settings.STATIC_ROOT, 'too close to camera.mp3')
sound_effect = pygame.mixer.Sound(sound_file_path0)
sound1 = pygame.mixer.Sound(sound_file_path1)
sound2 = pygame.mixer.Sound(sound_file_path2 )
sound3 = pygame.mixer.Sound(sound_file_path3)

sound1_played = False
sound2_played = False
sound3_played = False

def cleanup():
    global start_time, total_time, is_doing, sound1_played, sound2_played, sound3_played
    start_time = 0
    total_time = 0
    is_doing = False
    sound1_played = False
    sound2_played = False
    sound3_played = False

def get_pose_landmark_points():
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
hidden_landmarks = [0, 1, 2, 3, 4, 5, 6, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30]

def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f'{int(seconds):02d}'

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
        
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        
    if angle < 0:
        angle += 360
    return angle

def calculateAngle2(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Reflect the angle around the 180-degree axis
    angle = 360 - angle

    # Normalize the angle to be within [0, 360) range
    if angle >= 360:
        angle -= 360
    
    return angle

def calculateDistance(landmark1, landmark2):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)*1000

class VideoCamera:
    def __init__(self):
        print("Initializing VideoCamera...")
        self.video = cv2.VideoCapture(0)
        #self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def __del__(self):
        print("Releasing VideoCamera...")
        self.video.release()

    def detectPose(self):
        landmarks = []
        point = get_pose_landmark_points()
        distance_camera = 0
        right_shoulder_angle = 0
        frame = None  # Initialize frame as None
               
        success, frame = self.video.read()
        
        if not success:
            print("Failed to get frame from VideoCamera.")
            return None  # Return None if frame retrieval fails
            # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)
        
        screen_width = 1920  
        screen_height = 1080 

        # Resize the frame to fit the screen while maintaining the aspect ratio
        frame_height, frame_width, _ = frame.shape
        if frame_width / screen_width > frame_height / screen_height:
            scale = screen_width / frame_width
        else:
            scale = screen_height / frame_height
        frame = cv2.resize(frame, (int(frame_width * scale), int(frame_height * scale)))
        imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5).process(imageRGB)

        if results.pose_landmarks:
            print("Drawing landmarks on frame...")
            connections_without_hidden = [c for c in mp_pose.POSE_CONNECTIONS if c[0] not in hidden_landmarks and c[1] not in hidden_landmarks]
        
            mp_drawing.draw_landmarks(frame,
                                    landmark_list=results.pose_landmarks,
                                    connections=connections_without_hidden,
                                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 191, 0), thickness=5),
                                    landmark_drawing_spec=None,)
            
            for landmark in results.pose_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z))             

        return frame, distance_camera, right_shoulder_angle, landmarks  # Return all four values

def standingforwardbend(landmarks, output_image, display=False):
    
    global start_time, total_time, is_doing, sound1_played, sound2_played, sound3_played
    label = 'Unknow Pose'
    color = (255, 255, 255)
    color2 = (0, 0, 255)
    point = get_pose_landmark_points()
    finished = False
    
    right_knee_angle = calculateAngle(landmarks[point[24]],landmarks[point[26]],landmarks[point[28]])
    right_hip_angle = calculateAngle(landmarks[point[26]],landmarks[point[24]],landmarks[point[12]])
    left_knee_angle = calculateAngle(landmarks[point[23]],landmarks[point[25]],landmarks[point[27]])
    left_hip_angle = calculateAngle(landmarks[point[25]],landmarks[point[23]],landmarks[point[11]])
    
    # Calculate the distance between landmarks
    distance_camera = calculateDistance(landmarks[point[12]],landmarks[point[24]])
    
    if distance_camera >= 400:
        label = 'Too Close to Camera'
        color = (44,46,51)
        if not sound3_played:
            sound3.play()
            sound1_played = False
            sound2_played = False
            sound3_played = True
        
    else:
        if (165 <= right_knee_angle <= 195 and 95 >= right_hip_angle > 30) and (165 <= left_knee_angle <= 195 and 95 >= left_hip_angle > 30):
            label = 'correct pose'
            color = (0, 255, 0)
            if not sound1_played:
                sound1.play() 
                sound1_played = True
                sound2_played = False
                sound3_played = False
          
        elif (165 <= right_knee_angle <= 195 and right_hip_angle <= 30) and (165 <= left_knee_angle <= 195 and left_hip_angle <= 30): 
            label = 'correct! you so flexible'
            color = (0, 0, 255)
            if not sound1_played:
                sound1.play() 
                sound1_played = True
                sound2_played = False
                sound3_played = False

        elif (right_knee_angle < 165 and left_knee_angle < 165 and 95 >= right_hip_angle > 30) or (right_knee_angle < 165 and left_knee_angle < 165 and right_hip_angle <= 30):
            label = 'Make your legs tighter'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False

        elif 165 <= right_knee_angle <= 195 and 165 <= left_knee_angle <= 195 and right_hip_angle > 95 and left_hip_angle > 95:                                                                                                     
            label = 'Bend down more'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False

    x, y, w, h = 600, 1000, 600, 50
    padding = 10  
    margin = 15  

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    rect_width = text_width + 2 * (padding + margin)  
    rect_height = text_height + 2 * (padding + margin)  

    rect_x = x - margin
    rect_y = y - margin

    cv2.rectangle(output_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255,255,255), -1)

    text_x = rect_x + margin + padding
    text_y = rect_y + margin + padding

    cv2.putText(output_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(output_image, f'{int(left_knee_angle)}', (int(landmarks[point[25]][0]), int(landmarks[point[25]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(left_hip_angle)}', (int(landmarks[point[23]][0]), int(landmarks[point[23]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    
    knee_angle = int(landmarks[point[25]][0]+2), int(landmarks[point[25]][1]+2)
    hip_angle = int(landmarks[point[23]][0]+2), int(landmarks[point[23]][1]+2)         
    multiplier = -1
    
    
    count = ["{:02d}".format(num) for num in range(30, -1, -1)]

    if label == 'correct pose' or label == 'correct! you so flexible':
        if not is_doing:
            start_time = time() - total_time  
            is_doing = True
    else:
        if is_doing:
            is_doing = False
    
    if is_doing:
        elapsed_time = time() - start_time
        total_time = elapsed_time  
        
        if elapsed_time >= 30 and elapsed_time % 30 <= 0.1:
            sound_effect.play()
    else:
        elapsed_time = total_time
    
    formatted_time = format_time(elapsed_time)
    

    center_x, center_y = 1320, 90 
    radius = 50  
    cv2.circle(output_image, (center_x, center_y), radius, (255,255,255) , -1)
    
    if elapsed_time <= 31:
        cv2.putText(output_image, str(count[int(str(formatted_time))]), (1300, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    
    elif elapsed_time > 31:
        cv2.putText(output_image, 'End!', (1285, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        finished = True
        
    if display:      
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:,:,::-1]);plt.title('output image');plt.axis('off');
    else:
        return output_image, label, finished   
    
@gzip_page
def standingforwardbend_detection(request):
    cam = None  # Initialize cam variable outside of try block
    try:
        cam = VideoCamera()  # Instantiate cam

        # Function to generate frames
        def standingforwardbend_generate_frames(camera):
            try:
                while True:
                    frame_data = camera.detectPose()
                    if frame_data is None:
                        break
                    else:
                        frame, distance_camera, right_shoulder_angle, landmarks = frame_data

                        # Skip frames where landmarks list is empty
                        if not landmarks:
                            continue

                        # Apply standingforwardbend analysis to the frame
                        output_image, label, finished = standingforwardbend(landmarks, frame.copy())

                        # Overlay distance_camera onto the frame
                        cv2.putText(output_image, f"{int(distance_camera)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convert the frame to JPEG format
                        _, jpeg = cv2.imencode('.jpg', output_image)

                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
            finally:
                del camera

        # Return the streaming response
        return StreamingHttpResponse(standingforwardbend_generate_frames(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    
    finally:
        if cam is not None:
            del cam
        cleanup()  # Call cleanup function when exiting the view

def cobrastretch(landmarks, output_image, display=False):
    
    global start_time, total_time, is_doing, sound1_played, sound2_played, sound3_played
    label = 'Unknow Pose'
    color = (255, 255, 255)
    color2 = (0, 0, 255)
    point = get_pose_landmark_points()
    finished = False
    
    right_shoulder_angle = calculateAngle(landmarks[point[23]],landmarks[point[11]],landmarks[point[13]])
    right_knee_angle = calculateAngle(landmarks[point[23]],landmarks[point[25]],landmarks[point[27]])
    right_hip_angle = calculateAngle(landmarks[point[25]],landmarks[point[23]],landmarks[point[11]])
    right_ankle_angle = calculateAngle(landmarks[point[31]],landmarks[point[27]],landmarks[point[25]])
    
    # Calculate the distance between landmarks
    distance_camera = calculateDistance(landmarks[point[12]],landmarks[point[24]])
    
    if distance_camera >= 400:
        label = 'Too Close to Camera'
        color = (44,46,51)
        if not sound3_played:
            sound3.play()
            sound1_played = False
            sound2_played = False
            sound3_played = True
        
    else:
        if 10 <= right_shoulder_angle <= 70 and 200 <= right_hip_angle <= 280 and 160 <= right_knee_angle <= 185 and 180 >= right_ankle_angle >= 145:
            label = 'correct pose'
            color = (0, 255, 0)
            if not sound1_played:
                sound1.play() 
                sound1_played = True
                sound2_played = False
                sound3_played = False

        elif right_ankle_angle < 145 and 10 <= right_shoulder_angle <= 70 and 200 <= right_hip_angle <= 280 and 160 <= right_knee_angle <= 185:
            label = 'straighten your feet'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False

        elif right_knee_angle < 160 and 10 <= right_shoulder_angle <= 70 and 200 <= right_hip_angle <= 280 and 180 >= right_ankle_angle >= 145:
            label = 'Make your legs tighter'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False
                
        elif (right_hip_angle < 200 or 10 > right_shoulder_angle) and 160 <= right_knee_angle <= 185 and 180 >= right_ankle_angle >= 145 :
            label = 'straighten your back'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False
                
        else:
            label = 'Try again !'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False 
            
    #x, y, w, h = 300, 550, 300, 50
    x, y, w, h = 600, 1000, 600, 50
    padding = 10  
    margin = 15  

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    rect_width = text_width + 2 * (padding + margin)  
    rect_height = text_height + 2 * (padding + margin)  

    rect_x = x - margin
    rect_y = y - margin

    cv2.rectangle(output_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255,255,255), -1)

    text_x = rect_x + margin + padding
    text_y = rect_y + margin + padding

    cv2.putText(output_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(output_image, f'{int(right_knee_angle)}', (int(landmarks[point[25]][0]), int(landmarks[point[25]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(right_hip_angle)}', (int(landmarks[point[23]][0]), int(landmarks[point[23]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(right_shoulder_angle)}', (int(landmarks[point[11]][0]), int(landmarks[point[11]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(right_ankle_angle)}', (int(landmarks[point[27]][0]), int(landmarks[point[27]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    
    knee_angle = int(landmarks[point[25]][0]+2), int(landmarks[point[25]][1]+2)
    hip_angle = int(landmarks[point[23]][0]+2), int(landmarks[point[23]][1]+2)    
    shoulder_angle = int(landmarks[point[11]][0]+2), int(landmarks[point[11]][1]+2)      
    multiplier = -1
    
    count = ["{:02d}".format(num) for num in range(30, -1, -1)]

    if label == 'correct pose':
        if not is_doing:
            start_time = time() - total_time  
            is_doing = True
    else:
        if is_doing:
            is_doing = False
    
    if is_doing:
        elapsed_time = time() - start_time
        total_time = elapsed_time  
        
        if elapsed_time >= 30 and elapsed_time % 30 <= 0.1:
            sound_effect.play()
    else:
        elapsed_time = total_time
    
    formatted_time = format_time(elapsed_time)
    

    center_x, center_y = 1320, 90 
    radius = 50  
    cv2.circle(output_image, (center_x, center_y), radius, (255,255,255) , -1)
    
    if elapsed_time <= 31:
        cv2.putText(output_image, str(count[int(str(formatted_time))]), (1300, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    elif elapsed_time > 31:
        cv2.putText(output_image, 'End!', (1285, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        finished = True
        
    if display:      
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:,:,::-1]);plt.title('output image');plt.axis('off');
    else:
        return output_image, label, finished   
    
@gzip_page
def cobrastretch_detection(request):
    cam = None  # Initialize cam variable outside of try block
    try:
        cam = VideoCamera()  # Instantiate cam

        # Function to generate frames
        def cobrastretch_generate_frames(camera):
            try:
                while True:
                    frame_data = camera.detectPose()
                    if frame_data is None:
                        break
                    else:
                        frame, distance_camera, right_shoulder_angle, landmarks = frame_data

                        # Skip frames where landmarks list is empty
                        if not landmarks:
                            continue

                        # Apply standingforwardbend analysis to the frame
                        output_image, label, finished = cobrastretch(landmarks, frame.copy())

                        # Overlay distance_camera onto the frame
                        cv2.putText(output_image, f"{int(distance_camera)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convert the frame to JPEG format
                        _, jpeg = cv2.imencode('.jpg', output_image)

                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
            finally:
                del camera

        # Return the streaming response
        return StreamingHttpResponse(cobrastretch_generate_frames(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    
    finally:
        if cam is not None:
            del cam
        cleanup()  # Call cleanup function when exiting the view

def elbowplank(landmarks, output_image, display=False):
    
    global start_time, total_time, is_doing, sound1_played, sound2_played, sound3_played
    label = 'Unknow Pose'
    color = (255, 255, 255)
    color2 = (0, 0, 255)
    point = get_pose_landmark_points()
    finished = False
    
    right_shoulder_angle = calculateAngle(landmarks[point[23]],landmarks[point[11]],landmarks[point[13]])
    right_knee_angle = calculateAngle(landmarks[point[23]],landmarks[point[25]],landmarks[point[27]])
    right_hip_angle = calculateAngle(landmarks[point[25]],landmarks[point[23]],landmarks[point[11]])
    
    # Calculate the distance between landmarks
    distance_camera = calculateDistance(landmarks[point[12]],landmarks[point[24]])
    
    if distance_camera >= 400:
        label = 'Too Close to Camera'
        color = (44,46,51)
        if not sound3_played:
            sound3.play()
            sound1_played = False
            sound2_played = False
            sound3_played = True
        
    else:
        if 70 <= right_shoulder_angle <= 95 and 160 <= right_hip_angle <= 180 and 170 <= right_knee_angle <= 185:
            label = 'correct pose'
            color = (0, 255, 0)
            if not sound1_played:
                sound1.play() 
                sound1_played = True
                sound2_played = False
                sound3_played = False
        
        elif right_hip_angle < 160 and 70 <= right_shoulder_angle <= 95 and 170 <= right_knee_angle <= 185:
            label = 'Down your hips lower'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False

        elif right_hip_angle > 180 and 70 <= right_shoulder_angle <= 95 and 170 <= right_knee_angle <= 185:
            label = 'Raise your hips higher'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False

        elif right_knee_angle < 170 and 160 <= right_hip_angle <= 180 and 70 <= right_shoulder_angle <= 95:
            label = 'Make your legs tighter'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False
     
        else:
            label = 'Try again !'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False 
            
    #x, y, w, h = 300, 550, 300, 50
    x, y, w, h = 600, 1000, 600, 50
    padding = 10  
    margin = 15  

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    rect_width = text_width + 2 * (padding + margin)  
    rect_height = text_height + 2 * (padding + margin)  

    rect_x = x - margin
    rect_y = y - margin

    cv2.rectangle(output_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255,255,255), -1)

    text_x = rect_x + margin + padding
    text_y = rect_y + margin + padding

    cv2.putText(output_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(output_image, f'{int(right_knee_angle)}', (int(landmarks[point[25]][0]), int(landmarks[point[25]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(right_hip_angle)}', (int(landmarks[point[23]][0]), int(landmarks[point[23]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(right_shoulder_angle)}', (int(landmarks[point[11]][0]), int(landmarks[point[11]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    
    knee_angle = int(landmarks[point[25]][0]+2), int(landmarks[point[25]][1]+2)
    hip_angle = int(landmarks[point[23]][0]+2), int(landmarks[point[23]][1]+2)    
    shoulder_angle = int(landmarks[point[11]][0]+2), int(landmarks[point[11]][1]+2)      
    multiplier = -1
    
    count = ["{:02d}".format(num) for num in range(30, -1, -1)]

    if label == 'correct pose':
        if not is_doing:
            start_time = time() - total_time  
            is_doing = True
    else:
        if is_doing:
            is_doing = False
    
    if is_doing:
        elapsed_time = time() - start_time
        total_time = elapsed_time  
        
        if elapsed_time >= 30 and elapsed_time % 30 <= 0.1:
            sound_effect.play()
    else:
        elapsed_time = total_time
    
    formatted_time = format_time(elapsed_time)
    

    center_x, center_y = 1320, 90 
    radius = 50  
    cv2.circle(output_image, (center_x, center_y), radius, (255,255,255) , -1)
    
    if elapsed_time <= 31:
        cv2.putText(output_image, str(count[int(str(formatted_time))]), (1300, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    elif elapsed_time > 31:
        cv2.putText(output_image, 'End!', (1285, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        finished = True
        
    if display:      
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:,:,::-1]);plt.title('output image');plt.axis('off');
    else:
        return output_image, label, finished
    
@gzip_page
def elbowplank_detection(request):
    cam = None  # Initialize cam variable outside of try block
    try:
        cam = VideoCamera()  # Instantiate cam

        # Function to generate frames
        def elbowplank_generate_frames(camera):
            try:
                while True:
                    frame_data = camera.detectPose()
                    if frame_data is None:
                        break
                    else:
                        frame, distance_camera, right_shoulder_angle, landmarks = frame_data

                        # Skip frames where landmarks list is empty
                        if not landmarks:
                            continue

                        # Apply standingforwardbend analysis to the frame
                        output_image, label, finished = elbowplank(landmarks, frame.copy())

                        # Overlay distance_camera onto the frame
                        cv2.putText(output_image, f"{int(distance_camera)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convert the frame to JPEG format
                        _, jpeg = cv2.imencode('.jpg', output_image)

                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
            finally:
                del camera

        # Return the streaming response
        return StreamingHttpResponse(elbowplank_generate_frames(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    
    finally:
        if cam is not None:
            del cam
        cleanup()  # Call cleanup function when exiting the view

def childpose(landmarks, output_image, display=False):
    
    global start_time, total_time, is_doing, sound1_played, sound2_played, sound3_played
    label = 'Unknow Pose'
    color = (255, 255, 255)
    color2 = (0, 0, 255)
    point = get_pose_landmark_points()
    finished = False
    
    right_shoulder_angle = calculateAngle(landmarks[point[23]],landmarks[point[11]],landmarks[point[13]])
    right_knee_angle = calculateAngle(landmarks[point[23]],landmarks[point[25]],landmarks[point[27]])
    right_hip_angle = calculateAngle(landmarks[point[25]],landmarks[point[23]],landmarks[point[11]])
    right_ankle_angle = calculateAngle(landmarks[point[31]],landmarks[point[27]],landmarks[point[25]])
    
    # Calculate the distance between landmarks
    distance_camera = calculateDistance(landmarks[point[12]],landmarks[point[24]])
    
    if distance_camera >= 400:
        label = 'Too Close to Camera'
        color = (44,46,51)
        if not sound3_played:
            sound3.play()
            sound1_played = False
            sound2_played = False
            sound3_played = True
        
    else:
        if right_hip_angle <= 35 and right_knee_angle <= 35 and 180 >= right_ankle_angle >= 145:
            label = 'correct pose'
            color = (0, 255, 0)
            if not sound1_played:
                sound1.play() 
                sound1_played = True
                sound2_played = False
                sound3_played = False
        
        elif right_ankle_angle < 145 and 10 <= right_shoulder_angle <= 70 and 200 <= right_hip_angle <= 280 and 160 <= right_knee_angle <= 185:
            label = 'straighten your feet'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False

        elif right_knee_angle < 160 and 10 <= right_shoulder_angle <= 70 and 200 <= right_hip_angle <= 280 and 180 >= right_ankle_angle >= 145:
            label = 'Make your legs tighter'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False

        elif (right_hip_angle < 200 or 10 > right_shoulder_angle) and 160 <= right_knee_angle <= 185 and 180 >= right_ankle_angle >= 145 :
            label = 'straighten your back'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False
           
        else:
            label = 'Try again !'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False 
                 
    #x, y, w, h = 300, 550, 300, 50
    x, y, w, h = 600, 1000, 600, 50
    padding = 10  
    margin = 15  

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    rect_width = text_width + 2 * (padding + margin)  
    rect_height = text_height + 2 * (padding + margin)  

    rect_x = x - margin
    rect_y = y - margin

    cv2.rectangle(output_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255,255,255), -1)

    text_x = rect_x + margin + padding
    text_y = rect_y + margin + padding

    cv2.putText(output_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(output_image, f'{int(right_knee_angle)}', (int(landmarks[point[25]][0]), int(landmarks[point[25]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(right_hip_angle)}', (int(landmarks[point[23]][0]), int(landmarks[point[23]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(right_shoulder_angle)}', (int(landmarks[point[11]][0]), int(landmarks[point[11]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(right_ankle_angle)}', (int(landmarks[point[27]][0]), int(landmarks[point[27]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    
    knee_angle = int(landmarks[point[25]][0]+2), int(landmarks[point[25]][1]+2)
    hip_angle = int(landmarks[point[23]][0]+2), int(landmarks[point[23]][1]+2)    
    shoulder_angle = int(landmarks[point[11]][0]+2), int(landmarks[point[11]][1]+2)      
    multiplier = -1
    
    count = ["{:02d}".format(num) for num in range(30, -1, -1)]

    if label == 'correct pose':
        if not is_doing:
            start_time = time() - total_time  
            is_doing = True
    else:
        if is_doing:
            is_doing = False
    
    if is_doing:
        elapsed_time = time() - start_time
        total_time = elapsed_time  
        
        if elapsed_time >= 30 and elapsed_time % 30 <= 0.1:
            sound_effect.play()
    else:
        elapsed_time = total_time
    
    formatted_time = format_time(elapsed_time)
    

    center_x, center_y = 1320, 90 
    radius = 50  
    cv2.circle(output_image, (center_x, center_y), radius, (255,255,255) , -1)
    
    if elapsed_time <= 31:
        cv2.putText(output_image, str(count[int(str(formatted_time))]), (1300, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    elif elapsed_time > 31:
        cv2.putText(output_image, 'End!', (1285, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        finished = True
        
    if display:      
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:,:,::-1]);plt.title('output image');plt.axis('off');
    else:
        return output_image, label, finished

@gzip_page
def childpose_detection(request):
    cam = None  # Initialize cam variable outside of try block
    try:
        cam = VideoCamera()  # Instantiate cam

        # Function to generate frames
        def childpose_generate_frames(camera):
            try:
                while True:
                    frame_data = camera.detectPose()
                    if frame_data is None:
                        break
                    else:
                        frame, distance_camera, right_shoulder_angle, landmarks = frame_data

                        # Skip frames where landmarks list is empty
                        if not landmarks:
                            continue

                        # Apply standingforwardbend analysis to the frame
                        output_image, label, finished = childpose(landmarks, frame.copy())

                        # Overlay distance_camera onto the frame
                        cv2.putText(output_image, f"{int(distance_camera)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convert the frame to JPEG format
                        _, jpeg = cv2.imencode('.jpg', output_image)

                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
            finally:
                del camera

        # Return the streaming response
        return StreamingHttpResponse(childpose_generate_frames(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    
    finally:
        if cam is not None:
            del cam
        cleanup()  # Call cleanup function when exiting the view    

def neckstretch_R(landmarks, output_image, display=False):
    
    global start_time, total_time, is_doing, sound1_played, sound2_played, sound3_played
    label = 'Unknow Pose'
    color = (255, 255, 255)
    color2 = (0, 0, 255)
    point = get_pose_landmark_points()
    finished = False
    
    right_shoulder_angle = calculateAngle(landmarks[point[8]],landmarks[point[12]],landmarks[point[11]]) 
    left_shoulder_angle = calculateAngle2(landmarks[point[7]],landmarks[point[11]],landmarks[point[12]])
    
    # Calculate the distance between landmarks
    distance_camera = calculateDistance(landmarks[point[12]],landmarks[point[24]])
    
    if distance_camera >= 400:
        label = 'Too Close to Camera'
        color = (44,46,51)
        if not sound3_played:
            sound3.play()
            sound1_played = False
            sound2_played = False
            sound3_played = True
        
    else:
        
        if (25 <= right_shoulder_angle <= 50) :
            label = 'correct pose'
            color = (0, 255, 0)
            if not sound1_played:
                sound1.play() 
                sound1_played = True
                sound2_played = False
                sound3_played = False
                
        else:
            label = 'stretch more to the right !'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False  
                
    #x, y, w, h = 300, 550, 300, 50
    x, y, w, h = 600, 1000, 600, 50
    padding = 10  
    margin = 15  

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    rect_width = text_width + 2 * (padding + margin)  
    rect_height = text_height + 2 * (padding + margin)  

    rect_x = x - margin
    rect_y = y - margin

    cv2.rectangle(output_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255,255,255), -1)

    text_x = rect_x + margin + padding
    text_y = rect_y + margin + padding

    cv2.putText(output_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(output_image, f'{int(right_shoulder_angle)}', (int(landmarks[point[11]][0]), int(landmarks[point[11]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(left_shoulder_angle)}', (int(landmarks[point[12]][0]), int(landmarks[point[12]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)

    knee_angle = int(landmarks[point[25]][0]+2), int(landmarks[point[25]][1]+2)
    hip_angle = int(landmarks[point[23]][0]+2), int(landmarks[point[23]][1]+2)    
    shoulder_angle = int(landmarks[point[11]][0]+2), int(landmarks[point[11]][1]+2)      
    multiplier = -1
    
    count = ["{:02d}".format(num) for num in range(30, -1, -1)]

    if label == 'correct pose':
        if not is_doing:
            start_time = time() - total_time  
            is_doing = True
    else:
        if is_doing:
            is_doing = False
    
    if is_doing:
        elapsed_time = time() - start_time
        total_time = elapsed_time  
        
        if elapsed_time >= 30 and elapsed_time % 30 <= 0.1:
            sound_effect.play()
    else:
        elapsed_time = total_time
    
    formatted_time = format_time(elapsed_time)
    

    center_x, center_y = 1320, 90 
    radius = 50  
    cv2.circle(output_image, (center_x, center_y), radius, (255,255,255) , -1)
    
    if elapsed_time <= 31:
        cv2.putText(output_image, str(count[int(str(formatted_time))]), (1300, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    elif elapsed_time > 31:
        cv2.putText(output_image, 'End!', (1285, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        finished = True
        
    if display:      
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:,:,::-1]);plt.title('output image');plt.axis('off');
    else:
        return output_image, label, finished
  
@gzip_page
def neckstretch_R_detection(request):
    cam = None  # Initialize cam variable outside of try block
    try:
        cam = VideoCamera()  # Instantiate cam

        # Function to generate frames
        def neckstretch_R_generate_frames(camera):
            try:
                while True:
                    frame_data = camera.detectPose()
                    if frame_data is None:
                        break
                    else:
                        frame, distance_camera, right_shoulder_angle, landmarks = frame_data

                        # Skip frames where landmarks list is empty
                        if not landmarks:
                            continue

                        # Apply standingforwardbend analysis to the frame
                        output_image, label, finished = neckstretch_R(landmarks, frame.copy())

                        # Overlay distance_camera onto the frame
                        cv2.putText(output_image, f"{int(distance_camera)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convert the frame to JPEG format
                        _, jpeg = cv2.imencode('.jpg', output_image)

                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
            finally:
                del camera

        # Return the streaming response
        return StreamingHttpResponse(neckstretch_R_generate_frames(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    
    finally:
        if cam is not None:
            del cam
        cleanup()  # Call cleanup function when exiting the view    

def neckstretch_L(landmarks, output_image, display=False):
    
    global start_time, total_time, is_doing, sound1_played, sound2_played, sound3_played
    label = 'Unknow Pose'
    color = (255, 255, 255)
    color2 = (0, 0, 255)
    point = get_pose_landmark_points()
    finished = False
    
    right_shoulder_angle = calculateAngle(landmarks[point[8]],landmarks[point[12]],landmarks[point[11]]) 
    left_shoulder_angle = calculateAngle2(landmarks[point[7]],landmarks[point[11]],landmarks[point[12]])
    
    # Calculate the distance between landmarks
    distance_camera = calculateDistance(landmarks[point[12]],landmarks[point[24]])
    
    if distance_camera >= 400:
        label = 'Too Close to Camera'
        color = (44,46,51)
        if not sound3_played:
            sound3.play()
            sound1_played = False
            sound2_played = False
            sound3_played = True
        
    else:
        if (25 <= left_shoulder_angle <= 50) :
            label = 'correct pose'
            color = (0, 255, 0)
            if not sound1_played:
                sound1.play() 
                sound1_played = True
                sound2_played = False
                sound3_played = False
                
        else:
            label = 'stretch more to the left !'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False  
                
    #x, y, w, h = 300, 550, 300, 50
    x, y, w, h = 600, 1000, 600, 50
    padding = 10  
    margin = 15  

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    rect_width = text_width + 2 * (padding + margin)  
    rect_height = text_height + 2 * (padding + margin)  

    rect_x = x - margin
    rect_y = y - margin

    cv2.rectangle(output_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255,255,255), -1)

    text_x = rect_x + margin + padding
    text_y = rect_y + margin + padding

    cv2.putText(output_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(output_image, f'{int(right_shoulder_angle)}', (int(landmarks[point[11]][0]), int(landmarks[point[11]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(left_shoulder_angle)}', (int(landmarks[point[12]][0]), int(landmarks[point[12]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)

    knee_angle = int(landmarks[point[25]][0]+2), int(landmarks[point[25]][1]+2)
    hip_angle = int(landmarks[point[23]][0]+2), int(landmarks[point[23]][1]+2)    
    shoulder_angle = int(landmarks[point[11]][0]+2), int(landmarks[point[11]][1]+2)      
    multiplier = -1
    
    count = ["{:02d}".format(num) for num in range(30, -1, -1)]

    if label == 'correct pose':
        if not is_doing:
            start_time = time() - total_time  
            is_doing = True
    else:
        if is_doing:
            is_doing = False
    
    if is_doing:
        elapsed_time = time() - start_time
        total_time = elapsed_time  
        
        if elapsed_time >= 30 and elapsed_time % 30 <= 0.1:
            sound_effect.play()
    else:
        elapsed_time = total_time
    
    formatted_time = format_time(elapsed_time)
    

    center_x, center_y = 1320, 90 
    radius = 50  
    cv2.circle(output_image, (center_x, center_y), radius, (255,255,255) , -1)
    
    if elapsed_time <= 31:
        cv2.putText(output_image, str(count[int(str(formatted_time))]), (1300, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    elif elapsed_time > 31:
        cv2.putText(output_image, 'End!', (1285, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        finished = True
        
    if display:      
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:,:,::-1]);plt.title('output image');plt.axis('off');
    else:
        return output_image, label, finished
    
@gzip_page
def neckstretch_L_detection(request):
    cam = None  # Initialize cam variable outside of try block
    try:
        cam = VideoCamera()  # Instantiate cam

        # Function to generate frames
        def neckstretch_L_generate_frames(camera):
            try:
                while True:
                    frame_data = camera.detectPose()
                    if frame_data is None:
                        break
                    else:
                        frame, distance_camera, right_shoulder_angle, landmarks = frame_data

                        # Skip frames where landmarks list is empty
                        if not landmarks:
                            continue

                        # Apply standingforwardbend analysis to the frame
                        output_image, label, finished = neckstretch_L(landmarks, frame.copy())

                        # Overlay distance_camera onto the frame
                        cv2.putText(output_image, f"{int(distance_camera)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convert the frame to JPEG format
                        _, jpeg = cv2.imencode('.jpg', output_image)

                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
            finally:
                del camera

        # Return the streaming response
        return StreamingHttpResponse(neckstretch_L_generate_frames(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    
    finally:
        if cam is not None:
            del cam
        cleanup()  # Call cleanup function when exiting the view       

def shoulderstretch_R(landmarks, output_image, display=False):
    
    global start_time, total_time, is_doing, sound1_played, sound2_played, sound3_played
    label = 'Unknow Pose'
    color = (255, 255, 255)
    color2 = (0, 0, 255)
    point = get_pose_landmark_points()
    finished = False
    
    right_elbow_angle = calculateAngle(landmarks[point[15]],landmarks[point[13]],landmarks[point[11]]) 
    
    # Calculate the distance between landmarks
    distance_camera = calculateDistance(landmarks[point[12]],landmarks[point[24]])
    
    if distance_camera >= 400:
        label = 'Too Close to Camera'
        color = (44,46,51)
        if not sound3_played:
            sound3.play()
            sound1_played = False
            sound2_played = False
            sound3_played = True
        
    else:
        if 80 <= right_elbow_angle and right_elbow_angle <= 160 : 
            label = 'correct pose'
            color = (0, 255, 0)
            if not sound1_played:
                sound1.play() 
                sound1_played = True
                sound2_played = False
                sound3_played = False
                
        else:
            label = 'Try again !'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False   
 
    #x, y, w, h = 300, 550, 300, 50
    x, y, w, h = 600, 1000, 600, 50
    padding = 10  
    margin = 15  

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    rect_width = text_width + 2 * (padding + margin)  
    rect_height = text_height + 2 * (padding + margin)  

    rect_x = x - margin
    rect_y = y - margin

    cv2.rectangle(output_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255,255,255), -1)

    text_x = rect_x + margin + padding
    text_y = rect_y + margin + padding

    cv2.putText(output_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(output_image, f'{int(right_elbow_angle)}', (int(landmarks[point[13]][0]), int(landmarks[point[13]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    
    knee_angle = int(landmarks[point[25]][0]+2), int(landmarks[point[25]][1]+2)
    hip_angle = int(landmarks[point[23]][0]+2), int(landmarks[point[23]][1]+2)    
    shoulder_angle = int(landmarks[point[11]][0]+2), int(landmarks[point[11]][1]+2)      
    multiplier = -1
    
    count = ["{:02d}".format(num) for num in range(30, -1, -1)]

    if label == 'correct pose':
        if not is_doing:
            start_time = time() - total_time  
            is_doing = True
    else:
        if is_doing:
            is_doing = False
    
    if is_doing:
        elapsed_time = time() - start_time
        total_time = elapsed_time  
        
        if elapsed_time >= 30 and elapsed_time % 30 <= 0.1:
            sound_effect.play()
    else:
        elapsed_time = total_time
    
    formatted_time = format_time(elapsed_time)
    

    center_x, center_y = 1320, 90 
    radius = 50  
    cv2.circle(output_image, (center_x, center_y), radius, (255,255,255) , -1)
    
    if elapsed_time <= 31:
        cv2.putText(output_image, str(count[int(str(formatted_time))]), (1300, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    elif elapsed_time > 31:
        cv2.putText(output_image, 'End!', (1285, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        finished = True
        
    if display:      
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:,:,::-1]);plt.title('output image');plt.axis('off');
    else:
        return output_image, label, finished
    
@gzip_page
def shoulderstretch_R_detection(request):
    cam = None  # Initialize cam variable outside of try block
    try:
        cam = VideoCamera()  # Instantiate cam

        # Function to generate frames
        def shoulderstretch_R_generate_frames(camera):
            try:
                while True:
                    frame_data = camera.detectPose()
                    if frame_data is None:
                        break
                    else:
                        frame, distance_camera, right_shoulder_angle, landmarks = frame_data

                        # Skip frames where landmarks list is empty
                        if not landmarks:
                            continue

                        # Apply standingforwardbend analysis to the frame
                        output_image, label, finished = shoulderstretch_R(landmarks, frame.copy())

                        # Overlay distance_camera onto the frame
                        cv2.putText(output_image, f"{int(distance_camera)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convert the frame to JPEG format
                        _, jpeg = cv2.imencode('.jpg', output_image)

                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
            finally:
                del camera

        # Return the streaming response
        return StreamingHttpResponse(shoulderstretch_R_generate_frames(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    
    finally:
        if cam is not None:
            del cam
        cleanup()  # Call cleanup function when exiting the view      

def shoulderstretch_L(landmarks, output_image, display=False):
    
    global start_time, total_time, is_doing, sound1_played, sound2_played, sound3_played
    label = 'Unknow Pose'
    color = (255, 255, 255)
    color2 = (0, 0, 255)
    point = get_pose_landmark_points()
    finished = False
    
    left_elbow_angle = calculateAngle2(landmarks[point[16]],landmarks[point[14]],landmarks[point[12]]) 
    
    # Calculate the distance between landmarks
    distance_camera = calculateDistance(landmarks[point[12]],landmarks[point[24]])
    
    if distance_camera >= 400:
        label = 'Too Close to Camera'
        color = (44,46,51)
        if not sound3_played:
            sound3.play()
            sound1_played = False
            sound2_played = False
            sound3_played = True
        
    else:
        if 80 <= left_elbow_angle and left_elbow_angle <= 160: 
            label = 'correct pose'
            color = (0, 255, 0)
            if not sound1_played:
                sound1.play() 
                sound1_played = True
                sound2_played = False
                sound3_played = False

        else:
            label = 'Try again !'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False   
 
    #x, y, w, h = 300, 550, 300, 50
    x, y, w, h = 600, 1000, 600, 50
    padding = 10  
    margin = 15  

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    rect_width = text_width + 2 * (padding + margin)  
    rect_height = text_height + 2 * (padding + margin)  

    rect_x = x - margin
    rect_y = y - margin

    cv2.rectangle(output_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255,255,255), -1)

    text_x = rect_x + margin + padding
    text_y = rect_y + margin + padding

    cv2.putText(output_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(output_image, f'{int(left_elbow_angle)}', (int(landmarks[point[14]][0]), int(landmarks[point[14]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    
    knee_angle = int(landmarks[point[25]][0]+2), int(landmarks[point[25]][1]+2)
    hip_angle = int(landmarks[point[23]][0]+2), int(landmarks[point[23]][1]+2)    
    shoulder_angle = int(landmarks[point[11]][0]+2), int(landmarks[point[11]][1]+2)      
    multiplier = -1
    
    count = ["{:02d}".format(num) for num in range(30, -1, -1)]

    if label == 'correct pose':
        if not is_doing:
            start_time = time() - total_time  
            is_doing = True
    else:
        if is_doing:
            is_doing = False
    
    if is_doing:
        elapsed_time = time() - start_time
        total_time = elapsed_time  
        
        if elapsed_time >= 30 and elapsed_time % 30 <= 0.1:
            sound_effect.play()
    else:
        elapsed_time = total_time
    
    formatted_time = format_time(elapsed_time)
    

    center_x, center_y = 1320, 90 
    radius = 50  
    cv2.circle(output_image, (center_x, center_y), radius, (255,255,255) , -1)
    
    if elapsed_time <= 31:
        cv2.putText(output_image, str(count[int(str(formatted_time))]), (1300, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    elif elapsed_time > 31:
        cv2.putText(output_image, 'End!', (1285, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        finished = True
        
    if display:      
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:,:,::-1]);plt.title('output image');plt.axis('off');
    else:
        return output_image, label, finished
 
@gzip_page
def shoulderstretch_L_detection(request):
    cam = None  # Initialize cam variable outside of try block
    try:
        cam = VideoCamera()  # Instantiate cam

        # Function to generate frames
        def shoulderstretch_L_generate_frames(camera):
            try:
                while True:
                    frame_data = camera.detectPose()
                    if frame_data is None:
                        break
                    else:
                        frame, distance_camera, right_shoulder_angle, landmarks = frame_data

                        # Skip frames where landmarks list is empty
                        if not landmarks:
                            continue

                        # Apply standingforwardbend analysis to the frame
                        output_image, label, finished = shoulderstretch_L(landmarks, frame.copy())

                        # Overlay distance_camera onto the frame
                        cv2.putText(output_image, f"{int(distance_camera)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convert the frame to JPEG format
                        _, jpeg = cv2.imencode('.jpg', output_image)

                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
            finally:
                del camera

        # Return the streaming response
        return StreamingHttpResponse(shoulderstretch_L_generate_frames(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    
    finally:
        if cam is not None:
            del cam
        cleanup()  # Call cleanup function when exiting the view     

@gzip_page
def shoulderstretch_L_detection(request):
    cam = None  # Initialize cam variable outside of try block
    try:
        cam = VideoCamera()  # Instantiate cam

        # Function to generate frames
        def shoulderstretch_L_generate_frames(camera):
            try:
                while True:
                    frame_data = camera.detectPose()
                    if frame_data is None:
                        break
                    else:
                        frame, distance_camera, right_shoulder_angle, landmarks = frame_data

                        # Skip frames where landmarks list is empty
                        if not landmarks:
                            continue

                        # Apply standingforwardbend analysis to the frame
                        output_image, label, finished = shoulderstretch_L(landmarks, frame.copy())

                        # Overlay distance_camera onto the frame
                        cv2.putText(output_image, f"{int(distance_camera)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convert the frame to JPEG format
                        _, jpeg = cv2.imencode('.jpg', output_image)

                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
            finally:
                del camera

        # Return the streaming response
        return StreamingHttpResponse(shoulderstretch_L_generate_frames(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    
    finally:
        if cam is not None:
            del cam
        cleanup()  # Call cleanup function when exiting the view     

def overheadtricepstretch_R(landmarks, output_image, display=False):
    
    global start_time, total_time, is_doing, sound1_played, sound2_played, sound3_played
    label = 'Unknow Pose'
    color = (255, 255, 255)
    color2 = (0, 0, 255)
    point = get_pose_landmark_points()
    finished = False
    
    left_shoulder_angle = calculateAngle(landmarks[point[24]],landmarks[point[12]],landmarks[point[14]])
    left_elbow_angle = calculateAngle(landmarks[point[16]],landmarks[point[14]],landmarks[point[12]])
    
    right_shoulder_angle = calculateAngle2(landmarks[point[23]],landmarks[point[11]],landmarks[point[13]])
    right_elbow_angle = calculateAngle2(landmarks[point[15]],landmarks[point[13]],landmarks[point[11]])
     
    # Calculate the distance between landmarks
    distance_camera = calculateDistance(landmarks[point[12]],landmarks[point[24]])
    
    if distance_camera >= 400:
        label = 'Too Close to Camera'
        color = (44,46,51)
        if not sound3_played:
            sound3.play()
            sound1_played = False
            sound2_played = False
            sound3_played = True
        
    else:
        if (right_shoulder_angle >= 170  and right_elbow_angle <= 90 ) and (left_shoulder_angle >= 150  and left_elbow_angle <= 100 ):
            label = 'correct pose'
            color = (0, 255, 0)
            if not sound1_played:
                sound1.play() 
                sound1_played = True
                sound2_played = False
                sound3_played = False
        
        elif (right_shoulder_angle <= 170  and right_elbow_angle >= 90 ) or (left_shoulder_angle <= 150  and left_elbow_angle >= 100 ):
            label = 'stretch more'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False
                
        else:
            label = 'Try again !'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False   
            
    #x, y, w, h = 300, 550, 300, 50
    x, y, w, h = 600, 1000, 600, 50
    padding = 10  
    margin = 15  

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    rect_width = text_width + 2 * (padding + margin)  
    rect_height = text_height + 2 * (padding + margin)  

    rect_x = x - margin
    rect_y = y - margin

    cv2.rectangle(output_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255,255,255), -1)

    text_x = rect_x + margin + padding
    text_y = rect_y + margin + padding

    cv2.putText(output_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(output_image, f'{int(right_shoulder_angle)}', (int(landmarks[point[11]][0]), int(landmarks[point[11]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(right_elbow_angle)}', (int(landmarks[point[13]][0]), int(landmarks[point[13]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    
    cv2.putText(output_image, f'{int(left_shoulder_angle)}', (int(landmarks[point[12]][0]), int(landmarks[point[12]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(left_elbow_angle)}', (int(landmarks[point[14]][0]), int(landmarks[point[14]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    
    knee_angle = int(landmarks[point[25]][0]+2), int(landmarks[point[25]][1]+2)
    hip_angle = int(landmarks[point[23]][0]+2), int(landmarks[point[23]][1]+2)    
    #shoulder_angle = int(landmarks[point[11]][0]+2), int(landmarks[point[11]][1]+2)      
    multiplier = -1
    
    count = ["{:02d}".format(num) for num in range(30, -1, -1)]

    if label == 'correct pose':
        if not is_doing:
            start_time = time() - total_time  
            is_doing = True
    else:
        if is_doing:
            is_doing = False
    
    if is_doing:
        elapsed_time = time() - start_time
        total_time = elapsed_time  
        
        if elapsed_time >= 30 and elapsed_time % 30 <= 0.1:
            sound_effect.play()
    else:
        elapsed_time = total_time
    
    formatted_time = format_time(elapsed_time)
    

    center_x, center_y = 1320, 90 
    radius = 50  
    cv2.circle(output_image, (center_x, center_y), radius, (255,255,255) , -1)
    
    if elapsed_time <= 31:
        cv2.putText(output_image, str(count[int(str(formatted_time))]), (1300, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    elif elapsed_time > 31:
        cv2.putText(output_image, 'End!', (1285, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        finished = True
        
    if display:      
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:,:,::-1]);plt.title('output image');plt.axis('off');
    else:
        return output_image, label, finished
    
@gzip_page
def overheadtricepstretch_R_detection(request):
    cam = None  # Initialize cam variable outside of try block
    try:
        cam = VideoCamera()  # Instantiate cam

        # Function to generate frames
        def overheadtricepstretch_R_generate_frames(camera):
            try:
                while True:
                    frame_data = camera.detectPose()
                    if frame_data is None:
                        break
                    else:
                        frame, distance_camera, right_shoulder_angle, landmarks = frame_data

                        # Skip frames where landmarks list is empty
                        if not landmarks:
                            continue

                        # Apply standingforwardbend analysis to the frame
                        output_image, label, finished = overheadtricepstretch_R(landmarks, frame.copy())

                        # Overlay distance_camera onto the frame
                        cv2.putText(output_image, f"{int(distance_camera)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convert the frame to JPEG format
                        _, jpeg = cv2.imencode('.jpg', output_image)

                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
            finally:
                del camera

        # Return the streaming response
        return StreamingHttpResponse(overheadtricepstretch_R_generate_frames(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    
    finally:
        if cam is not None:
            del cam
        cleanup()  # Call cleanup function when exiting the view     

def overheadtricepstretch_L(landmarks, output_image, display=False):
    
    global start_time, total_time, is_doing, sound1_played, sound2_played, sound3_played
    label = 'Unknow Pose'
    color = (255, 255, 255)
    color2 = (0, 0, 255)
    point = get_pose_landmark_points()
    finished = False
    
    left_shoulder_angle = calculateAngle(landmarks[point[24]],landmarks[point[12]],landmarks[point[14]])
    left_elbow_angle = calculateAngle(landmarks[point[16]],landmarks[point[14]],landmarks[point[12]])
    
    right_shoulder_angle = calculateAngle2(landmarks[point[23]],landmarks[point[11]],landmarks[point[13]])
    right_elbow_angle = calculateAngle2(landmarks[point[15]],landmarks[point[13]],landmarks[point[11]])
    
    # Calculate the distance between landmarks
    distance_camera = calculateDistance(landmarks[point[12]],landmarks[point[24]])
    
    if distance_camera >= 400:
        label = 'Too Close to Camera'
        color = (44,46,51)
        if not sound3_played:
            sound3.play()
            sound1_played = False
            sound2_played = False
            sound3_played = True
        
    else:
        if (right_shoulder_angle >= 170  and right_elbow_angle <= 90 ) and (left_shoulder_angle >= 150  and left_elbow_angle <= 100 ):
            label = 'correct pose'
            color = (0, 255, 0)
            if not sound1_played:
                sound1.play() 
                sound1_played = True
                sound2_played = False
                sound3_played = False
        
        elif (right_shoulder_angle <= 170  and right_elbow_angle >= 90 ) or (left_shoulder_angle <= 150  and left_elbow_angle >= 100 ):
            label = 'stretch more'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False

        else:
            label = 'Try again !'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False   
            
    #x, y, w, h = 300, 550, 300, 50
    x, y, w, h = 600, 1000, 600, 50
    padding = 10  
    margin = 15  

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    rect_width = text_width + 2 * (padding + margin)  
    rect_height = text_height + 2 * (padding + margin)  

    rect_x = x - margin
    rect_y = y - margin

    cv2.rectangle(output_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255,255,255), -1)

    text_x = rect_x + margin + padding
    text_y = rect_y + margin + padding

    cv2.putText(output_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(output_image, f'{int(right_shoulder_angle)}', (int(landmarks[point[11]][0]), int(landmarks[point[11]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(right_elbow_angle)}', (int(landmarks[point[13]][0]), int(landmarks[point[13]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    
    cv2.putText(output_image, f'{int(left_shoulder_angle)}', (int(landmarks[point[12]][0]), int(landmarks[point[12]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(left_elbow_angle)}', (int(landmarks[point[14]][0]), int(landmarks[point[14]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    
    knee_angle = int(landmarks[point[25]][0]+2), int(landmarks[point[25]][1]+2)
    hip_angle = int(landmarks[point[23]][0]+2), int(landmarks[point[23]][1]+2)    
    #shoulder_angle = int(landmarks[point[11]][0]+2), int(landmarks[point[11]][1]+2)      
    multiplier = -1
    
    count = ["{:02d}".format(num) for num in range(30, -1, -1)]

    if label == 'correct pose':
        if not is_doing:
            start_time = time() - total_time  
            is_doing = True
    else:
        if is_doing:
            is_doing = False
    
    if is_doing:
        elapsed_time = time() - start_time
        total_time = elapsed_time  
        
        if elapsed_time >= 30 and elapsed_time % 30 <= 0.1:
            sound_effect.play()
    else:
        elapsed_time = total_time
    
    formatted_time = format_time(elapsed_time)
    

    center_x, center_y = 1320, 90 
    radius = 50  
    cv2.circle(output_image, (center_x, center_y), radius, (255,255,255) , -1)
    
    if elapsed_time <= 31:
        cv2.putText(output_image, str(count[int(str(formatted_time))]), (1300, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    elif elapsed_time > 31:
        cv2.putText(output_image, 'End!', (1285, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        finished = True
        
    if display:      
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:,:,::-1]);plt.title('output image');plt.axis('off');
    else:
        return output_image, label, finished
       
@gzip_page
def overheadtricepstretch_L_detection(request):
    cam = None  # Initialize cam variable outside of try block
    try:
        cam = VideoCamera()  # Instantiate cam

        # Function to generate frames
        def overheadtricepstretch_L_generate_frames(camera):
            try:
                while True:
                    frame_data = camera.detectPose()
                    if frame_data is None:
                        break
                    else:
                        frame, distance_camera, right_shoulder_angle, landmarks = frame_data

                        # Skip frames where landmarks list is empty
                        if not landmarks:
                            continue

                        # Apply standingforwardbend analysis to the frame
                        output_image, label, finished = overheadtricepstretch_L(landmarks, frame.copy())

                        # Overlay distance_camera onto the frame
                        cv2.putText(output_image, f"{int(distance_camera)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convert the frame to JPEG format
                        _, jpeg = cv2.imencode('.jpg', output_image)

                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
            finally:
                del camera

        # Return the streaming response
        return StreamingHttpResponse(overheadtricepstretch_L_generate_frames(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    
    finally:
        if cam is not None:
            del cam
        cleanup()  # Call cleanup function when exiting the view     

def quadricepstretch_L(landmarks, output_image, display=False):
    
    global start_time, total_time, is_doing, sound1_played, sound2_played, sound3_played
    label = 'Unknow Pose'
    color = (255, 255, 255)
    color2 = (0, 0, 255)
    point = get_pose_landmark_points()
    finished = False
    
    left_knee_angle = calculateAngle2(landmarks[point[24]],landmarks[point[26]],landmarks[point[28]])

    # Calculate the distance between landmarks
    distance_camera = calculateDistance(landmarks[point[12]],landmarks[point[24]])
    
    if distance_camera >= 400:
        label = 'Too Close to Camera'
        color = (44,46,51)
        if not sound3_played:
            sound3.play()
            sound1_played = False
            sound2_played = False
            sound3_played = True
        
    else:
        if (20 <= left_knee_angle <= 50):
            label = 'correct pose'
            color = (0, 255, 0)
            if not sound1_played:
                sound1.play() 
                sound1_played = True
                sound2_played = False
                sound3_played = False
        
        elif (left_knee_angle > 50): 
            label = 'make leg angle narrower'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False
                
        else:
            label = 'Try again !'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False   
        
    #x, y, w, h = 300, 550, 300, 50
    x, y, w, h = 600, 1000, 600, 50
    padding = 10  
    margin = 15  

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    rect_width = text_width + 2 * (padding + margin)  
    rect_height = text_height + 2 * (padding + margin)  

    rect_x = x - margin
    rect_y = y - margin

    cv2.rectangle(output_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255,255,255), -1)

    text_x = rect_x + margin + padding
    text_y = rect_y + margin + padding

    cv2.putText(output_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(output_image, f'{int(left_knee_angle)}', (int(landmarks[point[26]][0]), int(landmarks[point[26]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    
    knee_angle = int(landmarks[point[25]][0]+2), int(landmarks[point[25]][1]+2)
    hip_angle = int(landmarks[point[23]][0]+2), int(landmarks[point[23]][1]+2)    
    #shoulder_angle = int(landmarks[point[11]][0]+2), int(landmarks[point[11]][1]+2)      
    multiplier = -1
    
    count = ["{:02d}".format(num) for num in range(30, -1, -1)]

    if label == 'correct pose':
        if not is_doing:
            start_time = time() - total_time  
            is_doing = True
    else:
        if is_doing:
            is_doing = False
    
    if is_doing:
        elapsed_time = time() - start_time
        total_time = elapsed_time  
        
        if elapsed_time >= 30 and elapsed_time % 30 <= 0.1:
            sound_effect.play()
    else:
        elapsed_time = total_time
    
    formatted_time = format_time(elapsed_time)
    

    center_x, center_y = 1320, 90 
    radius = 50  
    cv2.circle(output_image, (center_x, center_y), radius, (255,255,255) , -1)
    
    if elapsed_time <= 31:
        cv2.putText(output_image, str(count[int(str(formatted_time))]), (1300, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    elif elapsed_time > 31:
        cv2.putText(output_image, 'End!', (1285, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        finished = True
        
    if display:      
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:,:,::-1]);plt.title('output image');plt.axis('off');
    else:
        return output_image, label, finished
    
@gzip_page
def quadricepstretch_L_detection(request):
    cam = None  # Initialize cam variable outside of try block
    try:
        cam = VideoCamera()  # Instantiate cam

        # Function to generate frames
        def quadricepstretch_L_generate_frames(camera):
            try:
                while True:
                    frame_data = camera.detectPose()
                    if frame_data is None:
                        break
                    else:
                        frame, distance_camera, right_shoulder_angle, landmarks = frame_data

                        # Skip frames where landmarks list is empty
                        if not landmarks:
                            continue

                        # Apply standingforwardbend analysis to the frame
                        output_image, label, finished = quadricepstretch_L(landmarks, frame.copy())

                        # Overlay distance_camera onto the frame
                        cv2.putText(output_image, f"{int(distance_camera)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convert the frame to JPEG format
                        _, jpeg = cv2.imencode('.jpg', output_image)

                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
            finally:
                del camera

        # Return the streaming response
        return StreamingHttpResponse(quadricepstretch_L_generate_frames(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    
    finally:
        if cam is not None:
            del cam
        cleanup()  # Call cleanup function when exiting the view  

def quadricepstretch_R(landmarks, output_image, display=False):
    
    global start_time, total_time, is_doing, sound1_played, sound2_played, sound3_played
    label = 'Unknow Pose'
    color = (255, 255, 255)
    color2 = (0, 0, 255)
    point = get_pose_landmark_points()
    finished = False
    
    right_knee_angle = calculateAngle(landmarks[point[23]],landmarks[point[25]],landmarks[point[27]])
    
    # Calculate the distance between landmarks
    distance_camera = calculateDistance(landmarks[point[12]],landmarks[point[24]])
    
    if distance_camera >= 400:
        label = 'Too Close to Camera'
        color = (44,46,51)
        if not sound3_played:
            sound3.play()
            sound1_played = False
            sound2_played = False
            sound3_played = True
        
    else:
        if (20 <= right_knee_angle <= 50):
            label = 'correct pose'
            color = (0, 255, 0)
            if not sound1_played:
                sound1.play() 
                sound1_played = True
                sound2_played = False
                sound3_played = False
        
        elif (right_knee_angle > 50):
            label = 'make leg angle narrower'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False
                
        else:
            label = 'Try again !'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False   
        
    #x, y, w, h = 300, 550, 300, 50
    x, y, w, h = 600, 1000, 600, 50
    padding = 10  
    margin = 15  

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    rect_width = text_width + 2 * (padding + margin)  
    rect_height = text_height + 2 * (padding + margin)  

    rect_x = x - margin
    rect_y = y - margin

    cv2.rectangle(output_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255,255,255), -1)

    text_x = rect_x + margin + padding
    text_y = rect_y + margin + padding

    cv2.putText(output_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(output_image, f'{int(right_knee_angle)}', (int(landmarks[point[25]][0]), int(landmarks[point[25]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    
    knee_angle = int(landmarks[point[25]][0]+2), int(landmarks[point[25]][1]+2)
    hip_angle = int(landmarks[point[23]][0]+2), int(landmarks[point[23]][1]+2)    
    #shoulder_angle = int(landmarks[point[11]][0]+2), int(landmarks[point[11]][1]+2)      
    multiplier = -1
    
    count = ["{:02d}".format(num) for num in range(30, -1, -1)]

    if label == 'correct pose':
        if not is_doing:
            start_time = time() - total_time  
            is_doing = True
    else:
        if is_doing:
            is_doing = False
    
    if is_doing:
        elapsed_time = time() - start_time
        total_time = elapsed_time  
        
        if elapsed_time >= 30 and elapsed_time % 30 <= 0.1:
            sound_effect.play()
    else:
        elapsed_time = total_time
    
    formatted_time = format_time(elapsed_time)
    

    center_x, center_y = 1320, 90 
    radius = 50  
    cv2.circle(output_image, (center_x, center_y), radius, (255,255,255) , -1)
    
    if elapsed_time <= 31:
        cv2.putText(output_image, str(count[int(str(formatted_time))]), (1300, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    elif elapsed_time > 31:
        cv2.putText(output_image, 'End!', (1285, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        finished = True
        
    if display:      
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:,:,::-1]);plt.title('output image');plt.axis('off');
    else:
        return output_image, label, finished
    
@gzip_page
def quadricepstretch_R_detection(request):
    cam = None  # Initialize cam variable outside of try block
    try:
        cam = VideoCamera()  # Instantiate cam

        # Function to generate frames
        def quadricepstretch_R_generate_frames(camera):
            try:
                while True:
                    frame_data = camera.detectPose()
                    if frame_data is None:
                        break
                    else:
                        frame, distance_camera, right_shoulder_angle, landmarks = frame_data

                        # Skip frames where landmarks list is empty
                        if not landmarks:
                            continue

                        # Apply standingforwardbend analysis to the frame
                        output_image, label, finished = quadricepstretch_R(landmarks, frame.copy())

                        # Overlay distance_camera onto the frame
                        cv2.putText(output_image, f"{int(distance_camera)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convert the frame to JPEG format
                        _, jpeg = cv2.imencode('.jpg', output_image)

                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
            finally:
                del camera

        # Return the streaming response
        return StreamingHttpResponse(quadricepstretch_R_generate_frames(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    
    finally:
        if cam is not None:
            del cam
        cleanup()  # Call cleanup function when exiting the view      
   
def reachingUp(landmarks, output_image, display=False):
    
    global start_time, total_time, is_doing, sound1_played, sound2_played, sound3_played
    label = 'Unknow Pose'
    color = (255, 255, 255)
    color2 = (0, 0, 255)
    point = get_pose_landmark_points()
    finished = False
    
    left_shoulder_angle = calculateAngle(landmarks[point[24]],landmarks[point[12]],landmarks[point[14]])
    left_elbow_angle = calculateAngle(landmarks[point[16]],landmarks[point[14]],landmarks[point[12]])
    
    right_shoulder_angle = calculateAngle2(landmarks[point[23]],landmarks[point[11]],landmarks[point[13]])
    right_elbow_angle = calculateAngle2(landmarks[point[15]],landmarks[point[13]],landmarks[point[11]])
     
    distance_shoulder= calculateDistance(landmarks[point[11]],landmarks[point[12]]) 
    distance_ankle= calculateDistance(landmarks[point[27]],landmarks[point[28]]) 
    
    # Calculate the distance between landmarks
    distance_camera = calculateDistance(landmarks[point[12]],landmarks[point[24]])
    
    if distance_camera >= 400:
        label = 'Too Close to Camera'
        color = (44,46,51)
        if not sound3_played:
            sound3.play()
            sound1_played = False
            sound2_played = False
            sound3_played = True
        
    else:
        if (right_shoulder_angle >= 170  and right_elbow_angle >= 180):
            label = 'correct pose'
            color = (0, 255, 0) 
            if not sound1_played:
                sound1.play() 
                sound1_played = True
                sound2_played = False
                sound3_played = False
        
        elif (right_shoulder_angle < 170  and right_elbow_angle < 180):
            label = 'stretch more'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False

        else:
            label = 'Try again !'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False   
        
    #x, y, w, h = 300, 550, 300, 50
    x, y, w, h = 600, 1000, 600, 50
    padding = 10  
    margin = 15  

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    rect_width = text_width + 2 * (padding + margin)  
    rect_height = text_height + 2 * (padding + margin)  

    rect_x = x - margin
    rect_y = y - margin

    cv2.rectangle(output_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255,255,255), -1)

    text_x = rect_x + margin + padding
    text_y = rect_y + margin + padding

    cv2.putText(output_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(output_image, f'{int(right_shoulder_angle)}', (int(landmarks[point[11]][0]), int(landmarks[point[11]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(right_elbow_angle)}', (int(landmarks[point[13]][0]), int(landmarks[point[13]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)

    knee_angle = int(landmarks[point[25]][0]+2), int(landmarks[point[25]][1]+2)
    hip_angle = int(landmarks[point[23]][0]+2), int(landmarks[point[23]][1]+2)    
    #shoulder_angle = int(landmarks[point[11]][0]+2), int(landmarks[point[11]][1]+2)      
    multiplier = -1
    
    count = ["{:02d}".format(num) for num in range(30, -1, -1)]

    if label == 'correct pose':
        if not is_doing:
            start_time = time() - total_time  
            is_doing = True
    else:
        if is_doing:
            is_doing = False
    
    if is_doing:
        elapsed_time = time() - start_time
        total_time = elapsed_time  
        
        if elapsed_time >= 30 and elapsed_time % 30 <= 0.1:
            sound_effect.play()
    else:
        elapsed_time = total_time
    
    formatted_time = format_time(elapsed_time)
    

    center_x, center_y = 1320, 90 
    radius = 50  
    cv2.circle(output_image, (center_x, center_y), radius, (255,255,255) , -1)
    
    if elapsed_time <= 31:
        cv2.putText(output_image, str(count[int(str(formatted_time))]), (1300, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    elif elapsed_time > 31:
        cv2.putText(output_image, 'End!', (1285, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        finished = True
        
    if display:      
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:,:,::-1]);plt.title('output image');plt.axis('off');
    else:
        return output_image, label, finished
  
@gzip_page
def reachingUp_detection(request):
    cam = None  # Initialize cam variable outside of try block
    try:
        cam = VideoCamera()  # Instantiate cam

        # Function to generate frames
        def reachingUp_generate_frames(camera):
            try:
                while True:
                    frame_data = camera.detectPose()
                    if frame_data is None:
                        break
                    else:
                        frame, distance_camera, right_shoulder_angle, landmarks = frame_data

                        # Skip frames where landmarks list is empty
                        if not landmarks:
                            continue

                        # Apply standingforwardbend analysis to the frame
                        output_image, label, finished = reachingUp(landmarks, frame.copy())

                        # Overlay distance_camera onto the frame
                        cv2.putText(output_image, f"{int(distance_camera)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convert the frame to JPEG format
                        _, jpeg = cv2.imencode('.jpg', output_image)

                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
            finally:
                del camera

        # Return the streaming response
        return StreamingHttpResponse(reachingUp_generate_frames(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    
    finally:
        if cam is not None:
            del cam
        cleanup()  # Call cleanup function when exiting the view      

def reachingDown(landmarks, output_image, display=False):
    
    global start_time, total_time, is_doing, sound1_played, sound2_played, sound3_played
    label = 'Unknow Pose'
    color = (255, 255, 255)
    color2 = (0, 0, 255)
    point = get_pose_landmark_points()
    finished = False
    
    right_knee_angle = calculateAngle(landmarks[point[24]],landmarks[point[26]],landmarks[point[28]])
    right_hip_angle = calculateAngle(landmarks[point[26]],landmarks[point[24]],landmarks[point[12]])
    left_knee_angle = calculateAngle(landmarks[point[23]],landmarks[point[25]],landmarks[point[27]])
    left_hip_angle = calculateAngle(landmarks[point[25]],landmarks[point[23]],landmarks[point[11]])
    
    # Calculate the distance between landmarks
    distance_camera = calculateDistance(landmarks[point[12]],landmarks[point[24]])
    
    if distance_camera >= 400:
        label = 'Too Close to Camera'
        color = (44,46,51)
        if not sound3_played:
            sound3.play()
            sound1_played = False
            sound2_played = False
            sound3_played = True
        
    else:
        if (165 <= right_knee_angle <= 195 and 95 >= right_hip_angle > 30) and (165 <= left_knee_angle <= 195 and 95 >= left_hip_angle > 30):
            label = 'correct pose'
            color = (0, 255, 0)
            if not sound1_played:
                sound1.play() 
                sound1_played = True
                sound2_played = False
                sound3_played = False

        elif (165 <= right_knee_angle <= 195 and 165 <= left_knee_angle <= 195 and right_hip_angle > 95 and left_hip_angle > 95):                                                                                                     
            label = 'Bend down more'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False
        
        else:
            label = 'Try again !'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False   
        
    #x, y, w, h = 300, 550, 300, 50
    x, y, w, h = 600, 1000, 600, 50
    padding = 10  
    margin = 15  

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    rect_width = text_width + 2 * (padding + margin)  
    rect_height = text_height + 2 * (padding + margin)  

    rect_x = x - margin
    rect_y = y - margin

    cv2.rectangle(output_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255,255,255), -1)

    text_x = rect_x + margin + padding
    text_y = rect_y + margin + padding

    cv2.putText(output_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(output_image, f'{int(left_knee_angle)}', (int(landmarks[point[25]][0]), int(landmarks[point[25]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(left_hip_angle)}', (int(landmarks[point[23]][0]), int(landmarks[point[23]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    
    knee_angle = int(landmarks[point[25]][0]+2), int(landmarks[point[25]][1]+2)
    hip_angle = int(landmarks[point[23]][0]+2), int(landmarks[point[23]][1]+2)    
    #shoulder_angle = int(landmarks[point[11]][0]+2), int(landmarks[point[11]][1]+2)      
    multiplier = -1
    
    count = ["{:02d}".format(num) for num in range(30, -1, -1)]

    if label == 'correct pose':
        if not is_doing:
            start_time = time() - total_time  
            is_doing = True
    else:
        if is_doing:
            is_doing = False
    
    if is_doing:
        elapsed_time = time() - start_time
        total_time = elapsed_time  
        
        if elapsed_time >= 30 and elapsed_time % 30 <= 0.1:
            sound_effect.play()
    else:
        elapsed_time = total_time
    
    formatted_time = format_time(elapsed_time)
    

    center_x, center_y = 1320, 90 
    radius = 50  
    cv2.circle(output_image, (center_x, center_y), radius, (255,255,255) , -1)
    
    if elapsed_time <= 31:
        cv2.putText(output_image, str(count[int(str(formatted_time))]), (1300, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    elif elapsed_time > 31:
        cv2.putText(output_image, 'End!', (1285, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        finished = True
        
    if display:      
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:,:,::-1]);plt.title('output image');plt.axis('off');
    else:
        return output_image, label, finished
    
@gzip_page
def reachingDown_detection(request):
    cam = None  # Initialize cam variable outside of try block
    try:
        cam = VideoCamera()  # Instantiate cam

        # Function to generate frames
        def reachingDown_generate_frames(camera):
            try:
                while True:
                    frame_data = camera.detectPose()
                    if frame_data is None:
                        break
                    else:
                        frame, distance_camera, right_shoulder_angle, landmarks = frame_data

                        # Skip frames where landmarks list is empty
                        if not landmarks:
                            continue

                        # Apply standingforwardbend analysis to the frame
                        output_image, label, finished = reachingDown(landmarks, frame.copy())

                        # Overlay distance_camera onto the frame
                        cv2.putText(output_image, f"{int(distance_camera)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convert the frame to JPEG format
                        _, jpeg = cv2.imencode('.jpg', output_image)

                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
            finally:
                del camera

        # Return the streaming response
        return StreamingHttpResponse(reachingDown_generate_frames(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    
    finally:
        if cam is not None:
            del cam
        cleanup()  # Call cleanup function when exiting the view         

def standingsidebend_R(landmarks, output_image, display=False):
    
    global start_time, total_time, is_doing, sound1_played, sound2_played, sound3_played
    label = 'Unknow Pose'
    color = (255, 255, 255)
    color2 = (0, 0, 255)
    point = get_pose_landmark_points()
    finished = False
    
    left_shoulder_angle = calculateAngle(landmarks[point[24]],landmarks[point[12]],landmarks[point[14]])
    right_shoulder_angle = calculateAngle2(landmarks[point[23]],landmarks[point[11]],landmarks[point[13]])

    # Calculate the distance between landmarks
    distance_camera = calculateDistance(landmarks[point[12]],landmarks[point[24]])
    
    if distance_camera >= 400:
        label = 'Too Close to Camera'
        color = (44,46,51)
        if not sound3_played:
            sound3.play()
            sound1_played = False
            sound2_played = False
            sound3_played = True
        
    else:
        if (140 <= right_shoulder_angle <= 160) and (195 <= left_shoulder_angle <= 230) :
            label = 'correct pose'
            color = (0, 255, 0)
            if not sound1_played:
                sound1.play() 
                sound1_played = True
                sound2_played = False
                sound3_played = False
        
        else:
            label = 'stretch more to the right !'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False
         
    #x, y, w, h = 300, 550, 300, 50
    x, y, w, h = 600, 1000, 600, 50
    padding = 10  
    margin = 15  

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    rect_width = text_width + 2 * (padding + margin)  
    rect_height = text_height + 2 * (padding + margin)  

    rect_x = x - margin
    rect_y = y - margin

    cv2.rectangle(output_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255,255,255), -1)

    text_x = rect_x + margin + padding
    text_y = rect_y + margin + padding

    cv2.putText(output_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(output_image, f'{int(right_shoulder_angle)}', (int(landmarks[point[11]][0]), int(landmarks[point[11]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(left_shoulder_angle)}', (int(landmarks[point[12]][0]), int(landmarks[point[12]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)

    knee_angle = int(landmarks[point[25]][0]+2), int(landmarks[point[25]][1]+2)
    hip_angle = int(landmarks[point[23]][0]+2), int(landmarks[point[23]][1]+2)    
    shoulder_angle = int(landmarks[point[11]][0]+2), int(landmarks[point[11]][1]+2)      
    multiplier = -1
    
    count = ["{:02d}".format(num) for num in range(30, -1, -1)]

    if label == 'correct pose':
        if not is_doing:
            start_time = time() - total_time  
            is_doing = True
    else:
        if is_doing:
            is_doing = False
    
    if is_doing:
        elapsed_time = time() - start_time
        total_time = elapsed_time  
        
        if elapsed_time >= 30 and elapsed_time % 30 <= 0.1:
            sound_effect.play()
    else:
        elapsed_time = total_time
    
    formatted_time = format_time(elapsed_time)
    

    center_x, center_y = 1320, 90 
    radius = 50  
    cv2.circle(output_image, (center_x, center_y), radius, (255,255,255) , -1)
    
    if elapsed_time <= 31:
        cv2.putText(output_image, str(count[int(str(formatted_time))]), (1300, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    elif elapsed_time > 31:
        cv2.putText(output_image, 'End!', (1285, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        finished = True
        
    if display:      
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:,:,::-1]);plt.title('output image');plt.axis('off');
    else:
        return output_image, label, finished

@gzip_page
def standingsidebend_R_detection(request):
    cam = None  # Initialize cam variable outside of try block
    try:
        cam = VideoCamera()  # Instantiate cam

        # Function to generate frames
        def standingsidebend_R_generate_frames(camera):
            try:
                while True:
                    frame_data = camera.detectPose()
                    if frame_data is None:
                        break
                    else:
                        frame, distance_camera, right_shoulder_angle, landmarks = frame_data

                        # Skip frames where landmarks list is empty
                        if not landmarks:
                            continue

                        # Apply standingforwardbend analysis to the frame
                        output_image, label, finished = standingsidebend_R(landmarks, frame.copy())

                        # Overlay distance_camera onto the frame
                        cv2.putText(output_image, f"{int(distance_camera)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convert the frame to JPEG format
                        _, jpeg = cv2.imencode('.jpg', output_image)

                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
            finally:
                del camera

        # Return the streaming response
        return StreamingHttpResponse(standingsidebend_R_generate_frames(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    
    finally:
        if cam is not None:
            del cam
        cleanup()  # Call cleanup function when exiting the view            

def standingsidebend_L(landmarks, output_image, display=False):
    
    global start_time, total_time, is_doing, sound1_played, sound2_played, sound3_played
    label = 'Unknow Pose'
    color = (255, 255, 255)
    color2 = (0, 0, 255)
    point = get_pose_landmark_points()
    finished = False
    
    left_shoulder_angle = calculateAngle(landmarks[point[24]],landmarks[point[12]],landmarks[point[14]])
    right_shoulder_angle = calculateAngle2(landmarks[point[23]],landmarks[point[11]],landmarks[point[13]])

    # Calculate the distance between landmarks
    distance_camera = calculateDistance(landmarks[point[12]],landmarks[point[24]])
    
    if distance_camera >= 400:
        label = 'Too Close to Camera'
        color = (44,46,51)
        if not sound3_played:
            sound3.play()
            sound1_played = False
            sound2_played = False
            sound3_played = True
        
    else:
        
        if (140 <= left_shoulder_angle <= 160) and (195 <= right_shoulder_angle <= 230) :
            label = 'correct pose'
            color = (0, 255, 0)
            if not sound1_played:
                sound1.play() 
                sound1_played = True
                sound2_played = False
                sound3_played = False
        
        else:
            label = 'stretch more to the left !'
            color = (0, 0, 255)
            if not sound2_played:
                sound2.play() 
                sound1_played = False
                sound2_played = True
                sound3_played = False
                
    #x, y, w, h = 300, 550, 300, 50
    x, y, w, h = 600, 1000, 600, 50
    padding = 10  
    margin = 15  

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    rect_width = text_width + 2 * (padding + margin)  
    rect_height = text_height + 2 * (padding + margin)  

    rect_x = x - margin
    rect_y = y - margin

    cv2.rectangle(output_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255,255,255), -1)

    text_x = rect_x + margin + padding
    text_y = rect_y + margin + padding

    cv2.putText(output_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(output_image, f'{int(right_shoulder_angle)}', (int(landmarks[point[11]][0]), int(landmarks[point[11]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
    cv2.putText(output_image, f'{int(left_shoulder_angle)}', (int(landmarks[point[12]][0]), int(landmarks[point[12]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
  
    knee_angle = int(landmarks[point[25]][0]+2), int(landmarks[point[25]][1]+2)
    hip_angle = int(landmarks[point[23]][0]+2), int(landmarks[point[23]][1]+2)    
    shoulder_angle = int(landmarks[point[11]][0]+2), int(landmarks[point[11]][1]+2)      
    multiplier = -1
    
    count = ["{:02d}".format(num) for num in range(30, -1, -1)]

    if label == 'correct pose':
        if not is_doing:
            start_time = time() - total_time  
            is_doing = True
    else:
        if is_doing:
            is_doing = False
    
    if is_doing:
        elapsed_time = time() - start_time
        total_time = elapsed_time  
        
        if elapsed_time >= 30 and elapsed_time % 30 <= 0.1:
            sound_effect.play()
    else:
        elapsed_time = total_time
    
    formatted_time = format_time(elapsed_time)
    

    center_x, center_y = 1320, 90 
    radius = 50  
    cv2.circle(output_image, (center_x, center_y), radius, (255,255,255) , -1)
    
    if elapsed_time <= 31:
        cv2.putText(output_image, str(count[int(str(formatted_time))]), (1300, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    elif elapsed_time > 31:
        cv2.putText(output_image, 'End!', (1285, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        finished = True
        
    if display:      
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:,:,::-1]);plt.title('output image');plt.axis('off');
    else:
        return output_image, label, finished
    
@gzip_page
def standingsidebend_L_detection(request):
    cam = None  # Initialize cam variable outside of try block
    try:
        cam = VideoCamera()  # Instantiate cam

        # Function to generate frames
        def standingsidebend_L_generate_frames(camera):
            try:
                while True:
                    frame_data = camera.detectPose()
                    if frame_data is None:
                        break
                    else:
                        frame, distance_camera, right_shoulder_angle, landmarks = frame_data

                        # Skip frames where landmarks list is empty
                        if not landmarks:
                            continue

                        # Apply standingforwardbend analysis to the frame
                        output_image, label, finished = standingsidebend_L(landmarks, frame.copy())

                        # Overlay distance_camera onto the frame
                        cv2.putText(output_image, f"{int(distance_camera)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convert the frame to JPEG format
                        _, jpeg = cv2.imencode('.jpg', output_image)

                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
            finally:
                del camera

        # Return the streaming response
        return StreamingHttpResponse(standingsidebend_L_generate_frames(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    
    finally:
        if cam is not None:
            del cam
        cleanup()  # Call cleanup function when exiting the view                    

