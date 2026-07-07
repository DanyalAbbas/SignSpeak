let isStreaming = false;
let socket = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initializeSocketConnection();
    updateUIState();
});

// Socket connection and handlers
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
        if (data && data.image) {
            const videoFeed = document.getElementById('videoFeed');
            videoFeed.src = 'data:image/jpeg;base64,' + data.image;
            videoFeed.style.display = 'block';
        }
    });
}

// UI update functions
function updateUIState() {
    const videoFeed = document.getElementById('videoFeed');
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const overlay = document.getElementById('cameraOffOverlay');
    
    videoFeed.style.display = isStreaming ? 'block' : 'none';
    overlay.style.display = isStreaming ? 'none' : 'flex';
    startButton.disabled = isStreaming;
    stopButton.disabled = !isStreaming;
    
    if (!isStreaming) {
        document.getElementById('translationText').textContent = 'No sign detected';
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
    
    // Add animation effect
    translationElement.style.animation = 'none';
    translationElement.offsetHeight; // Trigger reflow
    translationElement.style.animation = 'fadeIn 0.3s ease-in-out';
}

// Stream control functions
async function startStream() {
    if (!isStreaming) {
        try {
            // request permission and start local camera capture
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            startLocalCapture(stream);
            // notify server (keeps previous API semantics)
            await fetch('/start_stream', { method: 'POST' });
            isStreaming = true;
            updateUIState();
        } catch (error) {
            showError('Unable to access camera: ' + (error.message || error));
            console.error('Error starting stream:', error);
        }
    }
}

async function stopStream() {
    if (isStreaming) {
        try {
            await fetch('/stop_stream', { method: 'POST' });
            stopLocalCapture();
            isStreaming = false;
            document.getElementById('videoFeed').src = '';
            updateUIState();
        } catch (error) {
            showError('Network error occurred');
            console.error('Error:', error);
        }
    }
}

// Local capture helpers
let _localVideoElem = null;
let _captureCanvas = null;
let _captureIntervalId = null;

function startLocalCapture(stream) {
    if (!_localVideoElem) {
        _localVideoElem = document.createElement('video');
        _localVideoElem.setAttribute('playsinline', '');
        _localVideoElem.muted = true;
    }
    _localVideoElem.srcObject = stream;
    _localVideoElem.play();

    _captureCanvas = document.createElement('canvas');
    const ctx = _captureCanvas.getContext('2d');

    _localVideoElem.onloadedmetadata = () => {
        _captureCanvas.width = _localVideoElem.videoWidth || 640;
        _captureCanvas.height = _localVideoElem.videoHeight || 480;

        // capture at ~10 FPS
        _captureIntervalId = setInterval(() => {
            try {
                ctx.drawImage(_localVideoElem, 0, 0, _captureCanvas.width, _captureCanvas.height);
                const dataUrl = _captureCanvas.toDataURL('image/jpeg', 0.6);
                const base64 = dataUrl.split(',')[1];
                if (socket && socket.connected) {
                    socket.emit('frame', { image: base64 });
                }
            } catch (e) {
                console.error('capture error', e);
            }
        }, 100);
    };
}

function stopLocalCapture() {
    if (_captureIntervalId) {
        clearInterval(_captureIntervalId);
        _captureIntervalId = null;
    }
    if (_localVideoElem && _localVideoElem.srcObject) {
        const tracks = _localVideoElem.srcObject.getTracks();
        tracks.forEach(t => t.stop());
        _localVideoElem.srcObject = null;
    }
}

// Audio playback
function playAudio(audioUrl) {
    const audio = new Audio(audioUrl);
    audio.onplay = () => updateAudioStatus(true);
    audio.onended = () => updateAudioStatus(false);
    audio.onerror = () => {
        updateAudioStatus(false);
        showError('Failed to play audio');
    };
    
    audio.play().catch(error => {
        console.error('Audio playback error:', error);
        updateAudioStatus(false);
    });
}

// Error handling
function showError(message) {
    // You can implement a toast notification here
    console.error(message);
}