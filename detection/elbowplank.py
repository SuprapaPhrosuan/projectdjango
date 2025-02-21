#elbowplank.py
import base64
import cv2
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
import json
import mediapipe as mp
import detection.utills as u

mp_pose = mp.solutions.pose

hidden_landmarks = [0, 1, 2, 3, 4, 5, 6, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30]

class StreamConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_pose_landmarks = None

    async def connect(self):
        self.code = self.scope['url_route']['kwargs']['code']
        await self.channel_layer.group_add(self.code, self.channel_name)
        await self.accept()
        print("WebSocket connection accepted")

        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.code, self.channel_name)
        print("WebSocket connection closed")

        self.pose.close()

    async def receive(self, text_data):
        data = json.loads(text_data)
        frame_data = data.get('frame')
        
        if frame_data:
            decoded_data = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)
            frame = cv2.imdecode(decoded_data, cv2.IMREAD_COLOR)

            resized_frame = cv2.resize(frame, (640, 480))

            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                self.last_pose_landmarks = results.pose_landmarks
                point = u.get_pose_landmark_points()

                landmarks = [
                    (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                    for landmark in results.pose_landmarks.landmark
                ]

                connections_without_hidden = [
                    connection for connection in mp_pose.POSE_CONNECTIONS
                    if connection[0] not in hidden_landmarks and connection[1] not in hidden_landmarks
                ]

                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(
                    frame,
                    landmark_list=results.pose_landmarks,
                    connections=connections_without_hidden,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=5),
                    landmark_drawing_spec=None,
                )

                right_shoulder_angle = u.calculateAngle2(landmarks[point[24]],landmarks[point[12]],landmarks[point[14]])
                right_knee_angle = u.calculateAngle2(landmarks[point[24]],landmarks[point[26]],landmarks[point[28]])
                right_hip_angle = u.calculateAngle2(landmarks[point[26]],landmarks[point[24]],landmarks[point[12]])
    
                distance_camera = u.calculateDistance(landmarks[point[12]],landmarks[point[24]])
                color2 = (0, 0, 255)
                cv2.putText(frame, f'{int(right_knee_angle)}', (int(landmarks[point[26]][0]), int(landmarks[point[25]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
                cv2.putText(frame, f'{int(right_hip_angle)}', (int(landmarks[point[24]][0]), int(landmarks[point[23]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
                cv2.putText(frame, f'{int(right_shoulder_angle)}', (int(landmarks[point[12]][0]), int(landmarks[point[11]][1])), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)
                
                        
                if distance_camera >= 600:
                    label = 'Too Close to Camera'
                    color = (44,46,51)
                    
                else:
                    if 70 <= right_shoulder_angle <= 95 and 160 <= right_hip_angle <= 190 and 160 <= right_knee_angle <= 190:
                        label = 'Correct pose'
                        color = (0, 255, 0)
                    
                    elif 70 <= right_shoulder_angle <= 95 and right_hip_angle < 160 and 160 <= right_knee_angle <= 190:
                        label = 'Down your hips lower !' 
                        color = (0, 0, 255)

                    elif 70 <= right_shoulder_angle <= 95 and right_hip_angle < 160 and right_knee_angle < 160:
                        label = 'Make your legs tighter !'
                        color = (0, 0, 255)
                            
                    else:
                        label = 'Unknown' 
                        color = (0, 0, 255)

                padding_x = 20
                padding_y = 15  

                height, width, _ = frame.shape
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                label_x = int(width / 2) - int((label_width + padding_x * 2) / 2) 
                label_y = height - 20
                rect_top_left = (label_x, label_y - label_height - padding_y)
                rect_bottom_right = (label_x + label_width + padding_x * 2, label_y + padding_y)

                cv2.rectangle(frame, rect_top_left, rect_bottom_right, color, -1)
                cv2.putText(frame,label,(label_x + padding_x, label_y),cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255),2,cv2.LINE_AA)
                
             
                x1 = int(landmarks[point[12]][0])
                y1 = int(landmarks[point[12]][1])
                x2 = int(landmarks[point[24]][0])
                y2 = int(landmarks[point[24]][1])
                x3 = int(landmarks[point[28]][0])
                y3 = int(landmarks[point[28]][1])
                aux_image = np.zeros(frame.shape, np.uint8)
                cv2.line(aux_image, (x1, y1), (x2, y2), (255, 255, 255), 20)
                cv2.line(aux_image, (x2, y2), (x3, y3), (255, 255, 255), 20)
                cv2.line(aux_image, (x1, y1), (x3, y3), (255, 255, 255), 5)
                contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
                cv2.fillPoly(aux_image, pts=[contours], color=(0, 255, 0))
                frame = cv2.addWeighted(frame, 1, aux_image, 0.8, 0)
                cv2.circle(frame, (x1, y1), 6, (0, 255, 255), 4)
                cv2.circle(frame, (x2, y2), 6, (128, 0, 250), 4)
                cv2.circle(frame, (x3, y3), 6, (255, 191, 0), 4)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            await self.send(text_data=json.dumps({'frame': frame_base64}))




