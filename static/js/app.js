let isStreaming = false;
let socket = null;
let _localStream = null;
let _captureCanvas = null;
let _captureIntervalId = null;
let _hasProcessedFrame = false;
let _modelsReady = false;
let _frameInFlight = false;

const MAX_SEND_WIDTH = 480;

document.addEventListener('DOMContentLoaded', () => {
    initializeSocketConnection();
    updateUIState();
    pollModelStatus();
});

function initializeSocketConnection() {
    socket = io({
        transports: ['polling', 'websocket'],
        upgrade: true,
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
        if (data && data.text) {
            updateTranslation(data.text);
            if (data.audio) {
                playAudio(data.audio);
            }
        }
    });

    socket.on('processed_frame', (data) => {
        applyProcessedResult(data);
    });

    socket.on('connection_status', (data) => {
        if (data && typeof data.models_ready === 'boolean') {
            _modelsReady = data.models_ready;
            updateModelStatusUI();
        }
    });
}

function applyProcessedResult(data) {
    if (!data || !isStreaming) {
        return;
    }

    if (typeof data.models_ready === 'boolean') {
        _modelsReady = data.models_ready;
        updateModelStatusUI();
    }

    if (data.image) {
        const videoFeed = document.getElementById('videoFeed');
        videoFeed.src = 'data:image/jpeg;base64,' + data.image;
        videoFeed.style.display = 'block';
        _hasProcessedFrame = true;
    }

    if (data.translation && data.translation.text) {
        updateTranslation(data.translation.text);
        if (data.translation.audio) {
            playAudio(data.translation.audio);
        }
    }
}

async function pollModelStatus() {
    try {
        const res = await fetch('/health');
        const data = await res.json();
        _modelsReady = !!data.models_ready;
        updateModelStatusUI(data);
        if (!_modelsReady) {
            setTimeout(pollModelStatus, 2000);
        }
    } catch (e) {
        setTimeout(pollModelStatus, 3000);
    }
}

function updateModelStatusUI(data) {
    const el = document.getElementById('translationText');
    if (!el) {
        return;
    }
    if (data && data.models_error) {
        el.textContent = 'Model error: ' + data.models_error;
        return;
    }
    if (!_modelsReady) {
        el.textContent = 'Loading sign models… (first boot can take ~30s)';
        return;
    }
    if (!isStreaming) {
        el.textContent = 'No sign detected';
    }
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
        if (_modelsReady) {
            document.getElementById('translationText').textContent = 'No sign detected';
        }
    } else if (!_hasProcessedFrame) {
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
                width: { ideal: 640, max: 640 },
                height: { ideal: 480, max: 480 },
            },
            audio: false,
        });

        isStreaming = true;
        _hasProcessedFrame = false;
        updateUIState();
        if (!_modelsReady) {
            document.getElementById('translationText').textContent =
                'Camera on — waiting for models to finish loading…';
        }

        await startLocalCapture(_localStream);
        const res = await fetch('/start_stream', { method: 'POST' });
        const data = await res.json();
        _modelsReady = !!data.models_ready;
        updateModelStatusUI(data);
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

    await localVideo.play();
    await waitForVideoDimensions(localVideo);

    if (!_captureCanvas) {
        _captureCanvas = document.createElement('canvas');
    }

    const ctx = _captureCanvas.getContext('2d', { willReadFrequently: true });

    if (_captureIntervalId) {
        clearInterval(_captureIntervalId);
    }

    // HTTP processing is more reliable behind Cloudflare than Socket.IO payloads
    _captureIntervalId = setInterval(() => {
        captureAndSendFrame(localVideo, ctx);
    }, 200);
}

function captureAndSendFrame(localVideo, ctx) {
    if (!isStreaming || !localVideo.videoWidth || _frameInFlight) {
        return;
    }

    try {
        const srcW = localVideo.videoWidth;
        const srcH = localVideo.videoHeight;
        const scale = Math.min(1, MAX_SEND_WIDTH / srcW);
        const w = Math.max(1, Math.round(srcW * scale));
        const h = Math.max(1, Math.round(srcH * scale));

        if (_captureCanvas.width !== w || _captureCanvas.height !== h) {
            _captureCanvas.width = w;
            _captureCanvas.height = h;
        }

        ctx.drawImage(localVideo, 0, 0, w, h);
        const dataUrl = _captureCanvas.toDataURL('image/jpeg', 0.6);
        const base64 = dataUrl.split(',')[1];
        if (!base64) {
            return;
        }

        // Prefer HTTP — works through Cloudflare; Socket.IO as secondary
        sendFrameOverHttp(base64);
    } catch (e) {
        console.error('capture error', e);
    }
}

async function sendFrameOverHttp(base64) {
    _frameInFlight = true;
    try {
        const res = await fetch('/api/process_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64 }),
        });

        if (res.status === 429) {
            return;
        }

        if (!res.ok) {
            console.warn('process_frame failed', res.status);
            // Fallback to socket if HTTP path errors
            if (socket && socket.connected) {
                socket.emit('frame', { image: base64 });
            }
            return;
        }

        const data = await res.json();
        applyProcessedResult(data);
    } catch (e) {
        console.error('HTTP frame error', e);
        if (socket && socket.connected) {
            socket.emit('frame', { image: base64 });
        }
    } finally {
        _frameInFlight = false;
    }
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

    _frameInFlight = false;
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
