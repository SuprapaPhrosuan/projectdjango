#standingforwardbend.py
import base64
import cv2
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
import json
import asyncio
import mediapipe as mp
import detection.utills as u

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Define hidden landmarks
hidden_landmarks = [0, 1, 2, 3, 4, 5, 6, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30]

class StreamConsumerStandingforwardbend(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_pose_landmarks = None

    async def connect(self):
        self.code = self.scope['url_route']['kwargs']['code']
        await self.channel_layer.group_add(self.code, self.channel_name)
        await self.accept()
        print("WebSocket connection accepted")

        # Initialize MediaPipe Pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.code, self.channel_name)
        print("WebSocket connection closed")

        # Release MediaPipe resources
        self.pose.close()

    async def receive(self, text_data):
        data = json.loads(text_data)
        frame_data = data.get('frame')
        
        if frame_data:
            # Decode Base64 frame
            frame = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            # Reduce resolution to improve performance
            frame = cv2.resize(frame, (640, 480))

            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame with MediaPipe Pose
            results = self.pose.process(frame_rgb)

            # Check if landmarks are detected
            if results.pose_landmarks:
                self.last_pose_landmarks = results.pose_landmarks
            elif self.last_pose_landmarks:
                results.pose_landmarks = self.last_pose_landmarks

            # Draw landmarks on frame
            if results.pose_landmarks:
                # Extract landmarks as a list of tuples
                landmarks = [
                    (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                    for landmark in results.pose_landmarks.landmark
                ]

                # Filter connections to exclude hidden landmarks
                connections_without_hidden = [
                    connection for connection in mp_pose.POSE_CONNECTIONS
                    if connection[0] not in hidden_landmarks and connection[1] not in hidden_landmarks
                ]
                # Draw landmarks and filtered connections
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(
                    frame,
                    landmark_list=results.pose_landmarks,
                    connections=connections_without_hidden,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=5),
                    landmark_drawing_spec=None,
                )

            # Encode frame back to Base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Send processed frame back to client
            await self.send(text_data=json.dumps({'frame': frame_base64}))




