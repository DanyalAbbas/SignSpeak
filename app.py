import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
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
import logging
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'signspeak2025!'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
CORS(app)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=5 * 1024 * 1024,
)
_frame_lock = Lock()

# Global variables
prev_text = ""
last_time = time.time()
fps_calc = CvFpsCalc(buffer_len=10)

# Load the hand tracking model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
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



def text_to_speech_handler():
    try:
        os.makedirs('static/sounds', exist_ok=True)
        for label in keypoint_classifier_labels:
            if not os.path.exists(f"static/sounds/{label}.mp3"):
                tts = gTTS(label, lang='en')
                tts.save(f"static/sounds/{label}.mp3")
    except Exception as e:
        logger.error(f"Error in text_to_speech_handler: {str(e)}")

def decode_client_frame(b64):
    img_bytes = base64.b64decode(b64)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv.imdecode(arr, cv.IMREAD_COLOR)


def process_frame(frame):
    """Run hand tracking + sign classification.

    Returns {'label': str|None, 'landmarks': [[x, y], ...]|None}
    Landmarks are normalized 0-1 in the original (unflipped) camera frame
    so the browser overlay can share the video's CSS mirror.
    """
    try:
        frame = cv.flip(frame, 1)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks is None:
            return {'label': None, 'landmarks': None}

        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = [[1.0 - lm.x, lm.y] for lm in hand_landmarks.landmark]

        landmark_list = calc_landmark_list(frame, hand_landmarks)
        pre_processed_landmark_list = pre_process_landmark(landmark_list)
        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

        label = None
        if isinstance(hand_sign_id, tuple):
            sign_id, confidence = hand_sign_id
            accepted = confidence > 0.6
            if sign_id == 1 and confidence < 0.9:
                accepted = False
            if accepted and 0 <= sign_id < len(keypoint_classifier_labels):
                label = keypoint_classifier_labels[sign_id]
        elif 0 <= hand_sign_id < len(keypoint_classifier_labels):
            label = keypoint_classifier_labels[hand_sign_id]

        return {'label': label, 'landmarks': landmarks}
    except Exception as e:
        logger.error(f"Error in process_frame: {str(e)}")
        return {'label': None, 'landmarks': None}


def translation_payload(label):
    global prev_text, last_time
    if not label:
        return None

    speak = label != prev_text or (time.time() - last_time) >= 2
    if speak:
        prev_text = label
        last_time = time.time()

    return {
        'text': label,
        'audio': f'/static/sounds/{label}.mp3' if speak else None,
    }

# Flask routes
@app.after_request
def add_camera_headers(response):
    response.headers['Permissions-Policy'] = 'camera=(self), microphone=()'
    response.headers['Feature-Policy'] = "camera 'self'; microphone 'none'"
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin-allow-popups'
    if request.path.startswith('/static/'):
        # eventlet/gunicorn often logs/sends 0-byte static files with sendfile
        response.direct_passthrough = False
    return response

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'service': 'SignSpeak',
        'timestamp': time.time()
    })


@app.route('/video_feed')
def video_feed():
    # MJPEG streaming from server is not used on cloud deployments.
    # Clients should capture webcam locally and send frames via Socket.IO.
    return Response(status=404)


@app.route('/start_stream', methods=['POST'])
def start_stream():
    # For cloud deployment we don't open a local camera; client handles capture.
    return jsonify({'status': 'success'})


@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    return jsonify({'status': 'success'})


def run_classification(frame):
    # MediaPipe is CPU-bound; run it in a real thread so eventlet is not frozen.
    return eventlet.tpool.execute(process_frame, frame)


@app.route('/api/process_frame', methods=['POST'])
def api_process_frame():
    """HTTP path for sign detection. Used because Socket.IO frame payloads
    are often dropped in front of Render / Cloudflare."""
    data = request.get_json(silent=True) or {}
    b64 = data.get('image')
    if not b64:
        return jsonify({'error': 'missing image'}), 400

    if not _frame_lock.acquire(blocking=False):
        return jsonify({'busy': True}), 429

    try:
        frame = decode_client_frame(b64)
        if frame is None:
            return jsonify({'error': 'invalid image'}), 400

        result = run_classification(frame) or {}
        return jsonify({
            'translation': translation_payload(result.get('label')),
            'landmarks': result.get('landmarks'),
        })
    except Exception as e:
        logger.error(f"api_process_frame failed: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        _frame_lock.release()


@socketio.on('frame')
def handle_frame(data):
    """Fallback if the client still emits frames over Socket.IO."""
    if not _frame_lock.acquire(blocking=False):
        return

    try:
        b64 = data.get('image') if isinstance(data, dict) else None
        if not b64:
            return

        frame = decode_client_frame(b64)
        if frame is None:
            logger.warning('Failed to decode frame from client')
            return

        result = run_classification(frame) or {}
        payload = translation_payload(result.get('label'))
        emit('processed_hand', {
            'translation': payload,
            'landmarks': result.get('landmarks'),
        })
        if payload:
            emit('translation', payload)
    except Exception as e:
        logger.error(f"Error in frame handler: {str(e)}")
    finally:
        _frame_lock.release()

# SocketIO events
@socketio.on('connect')
def handle_connect():
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

# Helper functions
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
        return n / max_value if max_value != 0 else 0
    
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

# Run TTS generation at module level (for gunicorn)
text_to_speech_handler()

if __name__ == '__main__':
    # Start the Flask-SocketIO server
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
