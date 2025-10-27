from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
import cv2 as cv
import mediapipe as mp
import numpy as np
import csv
import copy
import itertools
from model import KeyPointClassifier
from utils import CvFpsCalc
from gtts import gTTS
import os
import time
from threading import Lock

app = Flask(__name__)
socketio = SocketIO(app)
thread = None
thread_lock = Lock()

# Global variables
camera = None
is_streaming = False
prev_text = ""
last_time = time.time()

# Load the hand tracking model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# Load the keypoint classifier
keypoint_classifier = KeyPointClassifier()

# Read labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

def init_camera():
    global camera
    if camera is None:
        camera = cv.VideoCapture(0)
        camera.set(cv.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, 540)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def text_to_speech_handler():
    os.makedirs('static/sounds', exist_ok=True)
    for label in keypoint_classifier_labels:
        if not os.path.exists(f"static/sounds/{label}.mp3"):
            tts = gTTS(label, lang='en')
            tts.save(f"static/sounds/{label}.mp3")

def process_frame(frame):
    global prev_text, last_time
    
    # Flip the frame horizontally
    frame = cv.flip(frame, 1)
    debug_image = copy.deepcopy(frame)
    
    # Convert to RGB for MediaPipe
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                            results.multi_handedness):
            # Calculate bounding box
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            
            # Calculate landmarks
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            
            # Preprocess landmarks
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            
            # Hand sign classification
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            
            if isinstance(hand_sign_id, tuple):
                if hand_sign_id[0] == 1 and hand_sign_id[1] < 0.9:
                    continue
                if hand_sign_id[1] <= 0.6:
                    continue
                hand_sign_id = hand_sign_id[0]
            
            # Draw landmarks and bounding box
            debug_image = draw_bounding_rect(True, debug_image, brect)
            debug_image = draw_landmarks(debug_image, landmark_list)
            
            # Get the translation text
            translation = keypoint_classifier_labels[hand_sign_id]
            debug_image = draw_info_text(debug_image, brect, handedness, translation)
            
            # Emit translation via WebSocket if it's different or enough time has passed
            if translation != prev_text or (time.time() - last_time) >= 2:
                audio_path = f"/static/sounds/{translation}.mp3"
                socketio.emit('translation', {
                    'text': translation,
                    'audio': audio_path
                })
                prev_text = translation
                last_time = time.time()
    
    # Add FPS info
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    debug_image = draw_info(debug_image, cvFpsCalc.get())
    
    return debug_image

def generate_frames():
    global is_streaming
    while is_streaming:
        success, frame = camera.read()
        if not success:
            break
        else:
            processed_frame = process_frame(frame)
            ret, buffer = cv.imencode('.jpg', processed_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global is_streaming
    init_camera()
    is_streaming = True
    return jsonify({'status': 'success'})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global is_streaming
    is_streaming = False
    release_camera()
    return jsonify({'status': 'success'})

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    
    # Convert to one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))
    
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Draw connections
        connections = [
            (2,3), (3,4),  # Thumb
            (5,6), (6,7), (7,8),  # Index finger
            (9,10), (10,11), (11,12),  # Middle finger
            (13,14), (14,15), (15,16),  # Ring finger
            (17,18), (18,19), (19,20),  # Little finger
            (0,1), (1,2), (2,5), (5,9),  # Palm
            (9,13), (13,17), (17,0)  # Palm
        ]
        
        for connection in connections:
            cv.line(image, tuple(landmark_point[connection[0]]), 
                   tuple(landmark_point[connection[1]]), (255,255,255), 2)
            cv.line(image, tuple(landmark_point[connection[0]]), 
                   tuple(landmark_point[connection[1]]), (0,0,0), 1)
        
        # Draw points
        for index, point in enumerate(landmark_point):
            radius = 8 if index in [4,8,12,16,20] else 5
            cv.circle(image, tuple(point), radius, (255,255,255), -1)
            cv.circle(image, tuple(point), radius, (0,0,0), 1)
    
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

def draw_info(image, fps):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image

if __name__ == '__main__':
    text_to_speech_handler()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)