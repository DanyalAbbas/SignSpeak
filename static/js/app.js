let isStreaming = false;
let socket = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initializeSocketConnection();
    updateUIState();
});

// Socket connection and handlers
function initializeSocketConnection() {
    socket = io();
    
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
            const response = await fetch('/start_stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            if (data.status === 'success') {
                isStreaming = true;
                document.getElementById('videoFeed').src = '/video_feed?' + new Date().getTime();
                updateUIState();
            } else {
                showError('Failed to start camera stream');
            }
        } catch (error) {
            showError('Network error occurred');
            console.error('Error:', error);
        }
    }
}

async function stopStream() {
    if (isStreaming) {
        try {
            const response = await fetch('/stop_stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            if (data.status === 'success') {
                isStreaming = false;
                document.getElementById('videoFeed').src = '';
                updateUIState();
            } else {
                showError('Failed to stop camera stream');
            }
        } catch (error) {
            showError('Network error occurred');
            console.error('Error:', error);
        }
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