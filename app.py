from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2 as cv
import numpy as np
import csv
import copy
import itertools
from utils import CvFpsCalc
from gtts import gTTS
import os
import time
from threading import Lock, Thread
import logging
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'signspeak2025!'
CORS(app)
# threading avoids eventlet monkey_patch conflicts with TensorFlow/MediaPipe
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Global variables
prev_text = ""
last_time = time.time()
fps_calc = CvFpsCalc(buffer_len=10)

_models_lock = Lock()
_hands = None
_keypoint_classifier = None
_models_ready = False
_models_error = None

with open('model/keypoint_classifier/keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]


def load_models():
    """Load MediaPipe + TFLite lazily so gunicorn can bind the port first."""
    global _hands, _keypoint_classifier, _models_ready, _models_error

    with _models_lock:
        if _models_ready:
            return True
        if _models_error is not None:
            return False

        try:
            logger.info('Loading MediaPipe and sign classifier...')
            import mediapipe as mp
            from model import KeyPointClassifier

            if not hasattr(mp, 'solutions'):
                raise RuntimeError(
                    'Installed mediapipe has no solutions API. '
                    'Pin mediapipe==0.10.21 in requirements.txt'
                )

            _hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            _keypoint_classifier = KeyPointClassifier()
            _models_ready = True
            logger.info('Models loaded successfully')
            return True
        except Exception as e:
            _models_error = str(e)
            logger.error(f'Failed to load models: {e}')
            return False


def text_to_speech_handler():
    try:
        os.makedirs('static/sounds', exist_ok=True)
        for label in keypoint_classifier_labels:
            if not os.path.exists(f"static/sounds/{label}.mp3"):
                tts = gTTS(label, lang='en')
                tts.save(f"static/sounds/{label}.mp3")
    except Exception as e:
        logger.error(f"Error in text_to_speech_handler: {str(e)}")


def process_frame(frame):
    global prev_text, last_time

    try:
        if not load_models():
            return frame

        # Flip the frame horizontally
        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame)

        # Convert to RGB for MediaPipe
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = _hands.process(frame_rgb)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                hand_sign_id = _keypoint_classifier(pre_processed_landmark_list)

                if isinstance(hand_sign_id, tuple):
                    if hand_sign_id[0] == 1 and hand_sign_id[1] < 0.9:
                        continue
                    if hand_sign_id[1] <= 0.6:
                        continue
                    hand_sign_id = hand_sign_id[0]

                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)

                translation = keypoint_classifier_labels[hand_sign_id]
                debug_image = draw_info_text(debug_image, brect, handedness, translation)

                if translation != prev_text or (time.time() - last_time) >= 2:
                    audio_path = f"/static/sounds/{translation}.mp3"
                    socketio.emit('translation', {
                        'text': translation,
                        'audio': audio_path
                    })
                    prev_text = translation
                    last_time = time.time()

        debug_image = draw_info(debug_image, fps_calc.get())
        return debug_image

    except Exception as e:
        logger.error(f"Error in process_frame: {str(e)}")
        return frame if frame is not None else None


@app.after_request
def add_camera_headers(response):
    response.headers['Permissions-Policy'] = 'camera=*, microphone=()'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin-allow-popups'
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'service': 'SignSpeak',
        'models_ready': _models_ready,
        'models_error': _models_error,
        'timestamp': time.time()
    })


@app.route('/video_feed')
def video_feed():
    return Response(status=404)


@app.route('/start_stream', methods=['POST'])
def start_stream():
    Thread(target=load_models, daemon=True).start()
    return jsonify({'status': 'success', 'models_ready': _models_ready})


@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    return jsonify({'status': 'success'})


@socketio.on('frame')
def handle_frame(data):
    """Receive a base64 JPEG frame from the client, process it, and return a processed JPEG."""
    try:
        b64 = data.get('image') if isinstance(data, dict) else None
        if not b64:
            return

        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv.imdecode(arr, cv.IMREAD_COLOR)
        if frame is None:
            logger.warning('Failed to decode frame from client')
            return

        processed = process_frame(frame)
        if processed is None:
            processed = frame

        ret, buf = cv.imencode('.jpg', processed, [int(cv.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            return

        out_b64 = base64.b64encode(buf).decode('utf-8')
        emit('processed_frame', {'image': out_b64})
    except Exception as e:
        logger.error(f"Error in frame handler: {str(e)}")


@socketio.on('connect')
def handle_connect():
    emit('connection_status', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')


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

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value if max_value != 0 else 0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        connections = [
            (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (0, 1), (1, 2), (2, 5), (5, 9),
            (9, 13), (13, 17), (17, 0)
        ]

        for connection in connections:
            cv.line(image, tuple(landmark_point[connection[0]]),
                    tuple(landmark_point[connection[1]]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[connection[0]]),
                    tuple(landmark_point[connection[1]]), (0, 0, 0), 1)

        for index, point in enumerate(landmark_point):
            radius = 8 if index in [4, 8, 12, 16, 20] else 5
            cv.circle(image, tuple(point), radius, (255, 255, 255), -1)
            cv.circle(image, tuple(point), radius, (0, 0, 0), 1)

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


# Lightweight startup only — do not block gunicorn port binding with TF/MediaPipe
Thread(target=text_to_speech_handler, daemon=True).start()
Thread(target=load_models, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
