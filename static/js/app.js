let isStreaming = false;
let socket = null;
let _localStream = null;
let _captureCanvas = null;
let _captureIntervalId = null;
let _hasProcessedFrame = false;

document.addEventListener('DOMContentLoaded', () => {
    initializeSocketConnection();
    updateUIState();
});

function initializeSocketConnection() {
    socket = io({
        transports: ['websocket', 'polling'],
    });

    socket.on('connect', () => {
        updateConnectionStatus(true);
        console.log('Connected to server');
    });

    socket.on('disconnect', () => {
        updateConnectionStatus(false);
        console.log('Disconnected from server');
    });

    socket.on('translation', (data) => {
        if (data.text) {
            updateTranslation(data.text);
            if (data.audio) {
                playAudio(data.audio);
            }
        }
    });

    // Receive processed frames from server (base64 jpeg)
    socket.on('processed_frame', (data) => {
        if (!data || !data.image || !isStreaming) {
            return;
        }
        const videoFeed = document.getElementById('videoFeed');
        videoFeed.src = 'data:image/jpeg;base64,' + data.image;
        videoFeed.style.display = 'block';
        _hasProcessedFrame = true;
    });
}

function updateUIState() {
    const videoFeed = document.getElementById('videoFeed');
    const localVideo = document.getElementById('localVideo');
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const overlay = document.getElementById('cameraOffOverlay');

    overlay.style.display = isStreaming ? 'none' : 'flex';
    localVideo.style.display = isStreaming ? 'block' : 'none';
    startButton.disabled = isStreaming;
    stopButton.disabled = !isStreaming;

    if (!isStreaming) {
        videoFeed.style.display = 'none';
        videoFeed.removeAttribute('src');
        _hasProcessedFrame = false;
        document.getElementById('translationText').textContent = 'No sign detected';
    } else if (!_hasProcessedFrame) {
        // Keep showing local preview until the first processed frame arrives
        videoFeed.style.display = 'none';
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

async function startStream() {
    if (isStreaming) {
        return;
    }

    try {
        _localStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user',
                width: { ideal: 640 },
                height: { ideal: 480 },
            },
            audio: false,
        });

        // Show local preview immediately, then begin sending frames
        isStreaming = true;
        _hasProcessedFrame = false;
        updateUIState();

        await startLocalCapture(_localStream);
        await fetch('/start_stream', { method: 'POST' });
    } catch (error) {
        stopLocalCapture();
        isStreaming = false;
        updateUIState();
        showError('Unable to access camera: ' + (error.message || error));
        console.error('Error starting stream:', error);
    }
}

async function stopStream() {
    if (!isStreaming) {
        return;
    }

    try {
        await fetch('/stop_stream', { method: 'POST' });
    } catch (error) {
        console.error('Error stopping stream:', error);
    }

    stopLocalCapture();
    isStreaming = false;
    updateUIState();
}

async function startLocalCapture(stream) {
    const localVideo = document.getElementById('localVideo');
    localVideo.srcObject = stream;
    localVideo.muted = true;
    localVideo.playsInline = true;

    // Wait until the video is actually playing with valid dimensions
    await localVideo.play();
    await waitForVideoDimensions(localVideo);

    if (!_captureCanvas) {
        _captureCanvas = document.createElement('canvas');
    }
    _captureCanvas.width = localVideo.videoWidth || 640;
    _captureCanvas.height = localVideo.videoHeight || 480;

    const ctx = _captureCanvas.getContext('2d', { willReadFrequently: true });

    if (_captureIntervalId) {
        clearInterval(_captureIntervalId);
    }

    // Capture at ~10 FPS and send frames to the server
    _captureIntervalId = setInterval(() => {
        if (!isStreaming || !localVideo.videoWidth) {
            return;
        }

        try {
            if (
                _captureCanvas.width !== localVideo.videoWidth ||
                _captureCanvas.height !== localVideo.videoHeight
            ) {
                _captureCanvas.width = localVideo.videoWidth;
                _captureCanvas.height = localVideo.videoHeight;
            }

            ctx.drawImage(localVideo, 0, 0, _captureCanvas.width, _captureCanvas.height);
            const dataUrl = _captureCanvas.toDataURL('image/jpeg', 0.7);
            const base64 = dataUrl.split(',')[1];

            if (socket && socket.connected && base64) {
                socket.emit('frame', { image: base64 });
            }
        } catch (e) {
            console.error('capture error', e);
        }
    }, 100);
}

function waitForVideoDimensions(video, timeoutMs = 5000) {
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
}

function playAudio(audioUrl) {
    const audio = new Audio(audioUrl);
    audio.onplay = () => updateAudioStatus(true);
    audio.onended = () => updateAudioStatus(false);
    audio.onerror = () => {
        updateAudioStatus(false);
        showError('Failed to play audio');
    };

    audio.play().catch((error) => {
        console.error('Audio playback error:', error);
        updateAudioStatus(false);
    });
}

function showError(message) {
    console.error(message);
}
