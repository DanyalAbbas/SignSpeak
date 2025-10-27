let isStreaming = false;

// WebSocket connection for real-time updates
const socket = new WebSocket('ws://' + window.location.host + '/ws');

socket.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.translation) {
        document.getElementById('translationText').textContent = data.translation;
        // Play audio if available
        if (data.audio) {
            playAudio(data.audio);
        }
    }
};

function startStream() {
    if (!isStreaming) {
        fetch('/start_stream', {method: 'POST'})
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    isStreaming = true;
                    document.getElementById('videoFeed').style.display = 'block';
                    document.getElementById('startButton').disabled = true;
                    document.getElementById('stopButton').disabled = false;
                }
            });
    }
}

function stopStream() {
    if (isStreaming) {
        fetch('/stop_stream', {method: 'POST'})
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    isStreaming = false;
                    document.getElementById('videoFeed').style.display = 'none';
                    document.getElementById('startButton').disabled = false;
                    document.getElementById('stopButton').disabled = true;
                    document.getElementById('translationText').textContent = '';
                }
            });
    }
}

function playAudio(audioUrl) {
    const audio = new Audio(audioUrl);
    audio.play();
}