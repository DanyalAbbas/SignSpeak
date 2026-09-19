let isStreaming = false;
let socket = null;
let _localStream = null;
let _captureCanvas = null;
let _captureIntervalId = null;
let _frameInFlight = false;

document.addEventListener('DOMContentLoaded', () => {
    initializeSocketConnection();
    updateUIState();
});

function initializeSocketConnection() {
    // Polling first is more reliable behind Render's proxy than opening
    // with websocket (which can "connect" then silently drop large frames).
    socket = io({
        transports: ['polling', 'websocket'],
        upgrade: true,
        reconnection: true,
        timeout: 20000,
    });

    socket.on('connect', () => {
        updateConnectionStatus(true);
        console.log('Connected to server');
        if (isStreaming) {
            setCameraError('');
        }
    });

    socket.on('disconnect', () => {
        updateConnectionStatus(false);
        console.log('Disconnected from server');
        if (isStreaming) {
            setCameraError('Camera is on. Reconnecting to the translator…');
        }
    });

    socket.on('connect_error', (error) => {
        console.error('Socket connection error:', error);
        if (isStreaming) {
            setCameraError('Camera is on, but the translator is not connected.');
        }
    });

    socket.on('translation', (data) => {
        if (data.text) {
            updateTranslation(data.text);
            if (data.audio) {
                playAudio(data.audio);
            }
        }
    });

    socket.on('processed_hand', (data) => {
        applyHandResult(data);
    });
}

function setCameraError(message) {
    const el = document.getElementById('cameraError');
    if (!el) {
        return;
    }
    el.textContent = message || '';
}

function updateUIState() {
    const localVideo = document.getElementById('localVideo');
    const handOverlay = document.getElementById('handOverlay');
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const overlay = document.getElementById('cameraOffOverlay');

    overlay.style.display = isStreaming ? 'none' : 'flex';
    localVideo.style.display = isStreaming ? 'block' : 'none';
    if (handOverlay) {
        handOverlay.style.display = isStreaming ? 'block' : 'none';
        if (!isStreaming) {
            clearHandOverlay();
        }
    }
    startButton.disabled = isStreaming;
    stopButton.disabled = !isStreaming;

    if (!isStreaming) {
        document.getElementById('translationText').textContent = 'No sign detected';
        setCameraError('');
    }
}

function updateConnectionStatus(connected) {
    const statusIcon = document.getElementById('connectionStatus');
    statusIcon.className = 'fas fa-circle ' + (connected ? 'connected' : 'disconnected');
}

function updateAudioStatus(active) {
    const audioIcon = document.getElementById('audioStatus');
    audioIcon.className = 'fas fa-microphone ' + (active ? 'active' : 'muted');
}

function updateTranslation(text) {
    const translationElement = document.getElementById('translationText');
    translationElement.textContent = text;

    translationElement.style.animation = 'none';
    translationElement.offsetHeight;
    translationElement.style.animation = 'fadeIn 0.3s ease-in-out';
}

async function getCameraStream() {
    const attempts = [
        {
            video: {
                facingMode: 'user',
                width: { ideal: 640 },
                height: { ideal: 480 },
            },
            audio: false,
        },
        { video: { facingMode: 'user' }, audio: false },
        { video: true, audio: false },
    ];

    let lastError;
    for (const constraints of attempts) {
        try {
            return await navigator.mediaDevices.getUserMedia(constraints);
        } catch (error) {
            lastError = error;
        }
    }

    throw lastError || new Error('Could not open camera');
}

async function startStream() {
    if (isStreaming) {
        return;
    }

    setCameraError('');

    try {
        if (!window.isSecureContext) {
            throw new Error('Camera requires HTTPS. Open the Render https:// URL.');
        }
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('This browser does not allow camera access.');
        }

        _localStream = await getCameraStream();

        // Keep the live webcam visible regardless of server/socket state.
        isStreaming = true;
        updateUIState();

        try {
            await startLocalCapture(_localStream);
        } catch (previewError) {
            console.warn('Camera preview warning:', previewError);
        }

        fetch('/start_stream', { method: 'POST' }).catch((error) => {
            console.warn('start_stream failed:', error);
        });
    } catch (error) {
        stopLocalCapture();
        isStreaming = false;
        updateUIState();
        setCameraError('Unable to access camera: ' + (error.message || error));
        console.error('Error starting stream:', error);
    }
}

async function stopStream() {
    if (!isStreaming) {
        return;
    }

    fetch('/stop_stream', { method: 'POST' }).catch((error) => {
        console.error('Error stopping stream:', error);
    });

    stopLocalCapture();
    isStreaming = false;
    updateUIState();
}

async function startLocalCapture(stream) {
    const localVideo = document.getElementById('localVideo');
    localVideo.srcObject = stream;
    localVideo.muted = true;
    localVideo.defaultMuted = true;
    localVideo.playsInline = true;
    localVideo.setAttribute('playsinline', 'true');
    localVideo.setAttribute('webkit-playsinline', 'true');
    localVideo.setAttribute('muted', '');
    localVideo.setAttribute('autoplay', '');

    try {
        await localVideo.play();
    } catch (error) {
        console.warn('video.play() warning:', error);
    }

    try {
        await waitForVideoDimensions(localVideo);
    } catch (error) {
        console.warn(error);
    }

    if (!_captureCanvas) {
        _captureCanvas = document.createElement('canvas');
    }

    const ctx = _captureCanvas.getContext('2d', { willReadFrequently: true });

    if (_captureIntervalId) {
        clearInterval(_captureIntervalId);
    }

    _captureIntervalId = setInterval(() => {
        if (!isStreaming || !localVideo.videoWidth || _frameInFlight) {
            return;
        }

        try {
            const maxWidth = 480;
            const scale = Math.min(1, maxWidth / localVideo.videoWidth);
            const width = Math.max(1, Math.round(localVideo.videoWidth * scale));
            const height = Math.max(1, Math.round(localVideo.videoHeight * scale));

            if (_captureCanvas.width !== width || _captureCanvas.height !== height) {
                _captureCanvas.width = width;
                _captureCanvas.height = height;
            }

            ctx.drawImage(localVideo, 0, 0, width, height);
            const dataUrl = _captureCanvas.toDataURL('image/jpeg', 0.6);
            const base64 = dataUrl.split(',')[1];
            if (base64) {
                sendFrameForTranslation(base64);
            }
        } catch (e) {
            console.error('capture error', e);
        }
    }, 250);
}

async function sendFrameForTranslation(base64) {
    _frameInFlight = true;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);
    try {
        const res = await fetch('/api/process_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64 }),
            signal: controller.signal,
        });

        if (res.status === 429) {
            try {
                const data = await res.json();
                if (data && data.models_ready === false) {
                    setCameraError('Camera on — loading sign models (first boot can take ~30s)…');
                }
            } catch (e) {
                /* ignore */
            }
            return;
        }
        if (res.ok) {
            setCameraError('');
        }
        if (!res.ok) {
            console.warn('process_frame HTTP ' + res.status);
            return;
        }

        const data = await res.json();
        applyHandResult(data);
    } catch (error) {
        if (error.name !== 'AbortError') {
            console.warn('HTTP frame failed:', error);
        }
    } finally {
        clearTimeout(timeoutId);
        _frameInFlight = false;
    }
}

function waitForVideoDimensions(video, timeoutMs = 8000) {
    if (video.videoWidth > 0 && video.videoHeight > 0) {
        return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
        const timeoutId = setTimeout(() => {
            cleanup();
            reject(new Error('Timed out waiting for camera video'));
        }, timeoutMs);

        const onReady = () => {
            if (video.videoWidth > 0 && video.videoHeight > 0) {
                cleanup();
                resolve();
            }
        };

        const cleanup = () => {
            clearTimeout(timeoutId);
            video.removeEventListener('loadedmetadata', onReady);
            video.removeEventListener('loadeddata', onReady);
            video.removeEventListener('playing', onReady);
        };

        video.addEventListener('loadedmetadata', onReady);
        video.addEventListener('loadeddata', onReady);
        video.addEventListener('playing', onReady);
        onReady();
    });
}

function stopLocalCapture() {
    if (_captureIntervalId) {
        clearInterval(_captureIntervalId);
        _captureIntervalId = null;
    }

    const localVideo = document.getElementById('localVideo');
    if (localVideo) {
        localVideo.pause();
        localVideo.srcObject = null;
    }

    if (_localStream) {
        _localStream.getTracks().forEach((track) => track.stop());
        _localStream = null;
    }

    _frameInFlight = false;
    clearHandOverlay();
}

function applyHandResult(data) {
    if (!isStreaming || !data) {
        return;
    }

    if (data.translation && data.translation.text) {
        updateTranslation(data.translation.text);
        if (data.translation.audio) {
            playAudio(data.translation.audio);
        }
    }

    drawHandSkeleton(data.landmarks, data.translation && data.translation.text);
}

const HAND_CONNECTIONS = [
    [2, 3], [3, 4],
    [5, 6], [6, 7], [7, 8],
    [9, 10], [10, 11], [11, 12],
    [13, 14], [14, 15], [15, 16],
    [17, 18], [18, 19], [19, 20],
    [0, 1], [1, 2], [2, 5], [5, 9],
    [9, 13], [13, 17], [17, 0],
];

function clearHandOverlay() {
    const canvas = document.getElementById('handOverlay');
    if (!canvas) {
        return;
    }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawHandSkeleton(landmarks, label) {
    const canvas = document.getElementById('handOverlay');
    const video = document.getElementById('localVideo');
    if (!canvas || !video) {
        return;
    }

    const width = video.videoWidth || canvas.clientWidth || 640;
    const height = video.videoHeight || canvas.clientHeight || 480;
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!landmarks || landmarks.length < 21) {
        return;
    }

    const points = landmarks.map(([x, y]) => [x * canvas.width, y * canvas.height]);

    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    HAND_CONNECTIONS.forEach(([a, b]) => {
        ctx.beginPath();
        ctx.moveTo(points[a][0], points[a][1]);
        ctx.lineTo(points[b][0], points[b][1]);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 4;
        ctx.stroke();
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.stroke();
    });

    points.forEach((point, index) => {
        const radius = [4, 8, 12, 16, 20].includes(index) ? 8 : 5;
        ctx.beginPath();
        ctx.arc(point[0], point[1], radius, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = '#000000';
        ctx.stroke();
    });

    if (label) {
        const [wx, wy] = points[0];
        ctx.font = '20px sans-serif';
        ctx.lineWidth = 4;
        ctx.strokeStyle = '#000000';
        ctx.strokeText(label, wx + 10, wy - 10);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, wx + 10, wy - 10);
    }
}

function playAudio(audioUrl) {
    const audio = new Audio(audioUrl);
    audio.onplay = () => updateAudioStatus(true);
    audio.onended = () => updateAudioStatus(false);
    audio.onerror = () => {
        updateAudioStatus(false);
        console.error('Failed to play audio');
    };

    audio.play().catch((error) => {
        console.error('Audio playback error:', error);
        updateAudioStatus(false);
    });
}
