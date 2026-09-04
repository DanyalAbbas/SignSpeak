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

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'signspeak2025!'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
CORS(app)

# Larger buffer so JPEG frames aren't silently dropped by Engine.IO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    max_http_buffer_size=10 * 1024 * 1024,
    ping_timeout=60,
    ping_interval=25,
)

prev_text = ""
last_time = time.time()
fps_calc = CvFpsCalc(buffer_len=10)

_models_lock = Lock()
_inference_lock = Lock()
_hands = None
_keypoint_classifier = None
_models_ready = False
_models_error = None
_models_loading = False
_load_started_at = 0.0

MIN_SIGN_CONFIDENCE = 0.50
MIN_THANK_YOU_CONFIDENCE = 0.70
LOAD_STALE_SECONDS = 180

with open('model/keypoint_classifier/keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]


def load_models(force=False):
    """Load MediaPipe + TFLite. Safe to call from any thread."""
    global _hands, _keypoint_classifier, _models_ready, _models_error
    global _models_loading, _load_started_at

    with _models_lock:
        if _models_ready and not force:
            return True

        # Recover if a previous background load hung / died mid-way
        if (
            _models_loading
            and not force
            and _load_started_at
            and (time.time() - _load_started_at) < LOAD_STALE_SECONDS
        ):
            return False

        _models_loading = True
        _load_started_at = time.time()
        _models_error = None

    try:
        logger.info('Loading MediaPipe and sign classifier...')
        import mediapipe as mp
        from model import KeyPointClassifier

        if not hasattr(mp, 'solutions'):
            raise RuntimeError(
                'Installed mediapipe has no solutions API. '
                'Pin mediapipe==0.10.21 in requirements.txt'
            )

        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        classifier = KeyPointClassifier()

        with _models_lock:
            _hands = hands
            _keypoint_classifier = classifier
            _models_ready = True
            _models_error = None
            _models_loading = False

        logger.info('Models loaded successfully')
        return True
    except Exception as e:
        with _models_lock:
            _models_error = str(e)
            _models_loading = False
            _models_ready = False
        logger.exception('Failed to load models')
        return False


def text_to_speech_handler():
    try:
        os.makedirs('static/sounds', exist_ok=True)
        for label in keypoint_classifier_labels:
            path = f"static/sounds/{label}.mp3"
            if not os.path.exists(path):
                tts = gTTS(label, lang='en')
                tts.save(path)
    except Exception as e:
        logger.error(f"Error in text_to_speech_handler: {str(e)}")


def decode_image_b64(b64):
    img_bytes = base64.b64decode(b64)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv.imdecode(arr, cv.IMREAD_COLOR)


def encode_image_b64(frame, quality=70):
    ret, buf = cv.imencode('.jpg', frame, [int(cv.IMWRITE_JPEG_QUALITY), quality])
    if not ret:
        return None
    return base64.b64encode(buf).decode('utf-8')


def process_frame(frame):
    global prev_text, last_time

    try:
        if not _models_ready and not load_models():
            return frame, None

        # Keep inference snappy on Render CPU
        h, w = frame.shape[:2]
        if w > 480:
            scale = 480.0 / w
            frame = cv.resize(frame, (480, int(h * scale)))

        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame)

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = _hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        translation = None
        confidence = None

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)

                prediction = _keypoint_classifier(pre_processed_landmark_list)
                if isinstance(prediction, tuple):
                    sign_id, confidence = prediction
                else:
                    sign_id, confidence = prediction, 1.0

                label = ""
                accepted = confidence >= MIN_SIGN_CONFIDENCE
                if accepted and sign_id == 1 and confidence < MIN_THANK_YOU_CONFIDENCE:
                    accepted = False

                if accepted and 0 <= sign_id < len(keypoint_classifier_labels):
                    label = keypoint_classifier_labels[sign_id]
                    translation = label

                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    label or f"{confidence:.0%}",
                )

                if translation and (
                    translation != prev_text or (time.time() - last_time) >= 1.5
                ):
                    prev_text = translation
                    last_time = time.time()
                    logger.info(
                        'Predicted sign=%s confidence=%.2f', translation, confidence
                    )

        debug_image = draw_info(debug_image, fps_calc.get())
        return debug_image, (
            {'text': translation, 'confidence': confidence} if translation else None
        )

    except Exception as e:
        logger.error(f"Error in process_frame: {str(e)}")
        return (frame if frame is not None else None), None


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
        'models_loading': _models_loading,
        'models_error': _models_error,
        'labels': keypoint_classifier_labels,
        'timestamp': time.time(),
    })


@app.route('/video_feed')
def video_feed():
    return Response(status=404)


@app.route('/start_stream', methods=['POST'])
def start_stream():
    if not _models_ready:
        Thread(target=load_models, daemon=True).start()
    return jsonify({
        'status': 'success',
        'models_ready': _models_ready,
        'models_loading': _models_loading,
        'models_error': _models_error,
    })


@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    return jsonify({'status': 'success'})


@app.route('/api/process_frame', methods=['POST'])
def api_process_frame():
    """HTTP fallback for Cloudflare / proxies that break Socket.IO binary-ish traffic."""
    data = request.get_json(silent=True) or {}
    b64 = data.get('image')
    if not b64:
        return jsonify({'error': 'missing image'}), 400

    if not _inference_lock.acquire(blocking=False):
        return jsonify({
            'busy': True,
            'models_ready': _models_ready,
        }), 429

    try:
        frame = decode_image_b64(b64)
        if frame is None:
            return jsonify({'error': 'invalid image'}), 400

        if not _models_ready:
            Thread(target=load_models, daemon=True).start()
            out_b64 = encode_image_b64(frame)
            return jsonify({
                'image': out_b64,
                'models_ready': False,
                'translation': None,
            })

        processed, prediction = process_frame(frame)
        if processed is None:
            processed = frame

        out_b64 = encode_image_b64(processed)
        payload = {
            'image': out_b64,
            'models_ready': True,
            'translation': None,
        }
        if prediction and prediction.get('text'):
            text = prediction['text']
            payload['translation'] = {
                'text': text,
                'audio': f'/static/sounds/{text}.mp3',
                'confidence': prediction.get('confidence'),
            }
        return jsonify(payload)
    except Exception as e:
        logger.exception('api_process_frame failed')
        return jsonify({'error': str(e)}), 500
    finally:
        _inference_lock.release()


@socketio.on('frame')
def handle_frame(data):
    if not _inference_lock.acquire(blocking=False):
        return

    try:
        b64 = data.get('image') if isinstance(data, dict) else None
        if not b64:
            return

        frame = decode_image_b64(b64)
        if frame is None:
            logger.warning('Failed to decode frame from client')
            return

        if not _models_ready:
            out_b64 = encode_image_b64(frame)
            if out_b64:
                emit('processed_frame', {'image': out_b64, 'models_ready': False})
            return

        processed, prediction = process_frame(frame)
        if processed is None:
            processed = frame

        out_b64 = encode_image_b64(processed)
        if not out_b64:
            return

        emit('processed_frame', {'image': out_b64, 'models_ready': True})
        if prediction and prediction.get('text'):
            text = prediction['text']
            emit('translation', {
                'text': text,
                'audio': f'/static/sounds/{text}.mp3',
                'confidence': float(prediction['confidence'] or 0),
            })
    except Exception as e:
        logger.error(f"Error in frame handler: {str(e)}")
    finally:
        _inference_lock.release()


@socketio.on('connect')
def handle_connect():
    emit('connection_status', {
        'status': 'connected',
        'models_ready': _models_ready,
    })


@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(
            landmark_array, [np.array((landmark_x, landmark_y))], axis=0
        )

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
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list))) or 1

    return [n / max_value for n in temp_landmark_list]


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        connections = [
            (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (0, 1), (1, 2), (2, 5), (5, 9),
            (9, 13), (13, 17), (17, 0),
        ]

        for a, b in connections:
            cv.line(image, tuple(landmark_point[a]), tuple(landmark_point[b]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[a]), tuple(landmark_point[b]),
                    (0, 0, 0), 1)

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


Thread(target=text_to_speech_handler, daemon=True).start()
Thread(target=load_models, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
