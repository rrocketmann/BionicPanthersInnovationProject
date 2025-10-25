// Connect to Socket.IO
const socket = io();

// DOM Elements
const uploadSection = document.getElementById('upload-section');
const processingSection = document.getElementById('processing-section');
const completeSection = document.getElementById('complete-section');
const errorSection = document.getElementById('error-section');

const uploadArea = document.getElementById('upload-area');
const videoInput = document.getElementById('video-input');
const fpsInput = document.getElementById('fps-input');
const selectedFileDiv = document.getElementById('selected-file');
const filenameSpan = document.getElementById('filename');
const startBtn = document.getElementById('start-btn');

const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const statusText = document.getElementById('status-text');

const downloadBtn = document.getElementById('download-btn');
const resetBtn = document.getElementById('reset-btn');
const retryBtn = document.getElementById('retry-btn');
const errorMessage = document.getElementById('error-message');

// State
let selectedFile = null;
let sessionId = null;
let downloadUrl = null;

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// File input handler
videoInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Handle file selection
function handleFileSelect(file) {
    // Validate file type
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please select a video file (MP4, AVI, MOV, MKV, WEBM).');
        return;
    }

    // Validate file size (500MB)
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File size exceeds 500MB limit.');
        return;
    }

    selectedFile = file;
    filenameSpan.textContent = file.name;
    selectedFileDiv.style.display = 'block';
}

// Start processing
startBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const fps = parseInt(fpsInput.value);
    if (isNaN(fps) || fps < 1 || fps > 120) {
        showError('Please enter a valid FPS value (1-120).');
        return;
    }

    // Upload file
    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('fps', fps);

    try {
        showSection('processing');
        updateProgress(0, 'Uploading video...');

        const uploadResponse = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const uploadData = await uploadResponse.json();

        if (!uploadResponse.ok) {
            throw new Error(uploadData.error || 'Upload failed');
        }

        sessionId = uploadData.session_id;

        // Start processing
        const processResponse = await fetch(`/process/${sessionId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ fps })
        });

        const processData = await processResponse.json();

        if (!processResponse.ok) {
            throw new Error(processData.error || 'Processing failed');
        }

        updateProgress(5, 'Processing started...');

    } catch (error) {
        showError(error.message);
    }
});

// Socket.IO event handlers
socket.on('progress', (data) => {
    if (data.session_id === sessionId) {
        updateProgress(data.percentage, data.status);

        if (data.percentage === 100 && data.download_url) {
            downloadUrl = data.download_url;
            showSection('complete');
        }
    }
});

socket.on('error', (data) => {
    if (!data.session_id || data.session_id === sessionId) {
        showError(data.message);
    }
});

// Download button
downloadBtn.addEventListener('click', () => {
    if (downloadUrl) {
        window.location.href = downloadUrl;
    }
});

// Reset button
resetBtn.addEventListener('click', () => {
    selectedFile = null;
    sessionId = null;
    downloadUrl = null;
    videoInput.value = '';
    selectedFileDiv.style.display = 'none';
    updateProgress(0, '');
    showSection('upload');
});

// Retry button
retryBtn.addEventListener('click', () => {
    showSection('upload');
});

// Helper functions
function showSection(section) {
    uploadSection.style.display = 'none';
    processingSection.style.display = 'none';
    completeSection.style.display = 'none';
    errorSection.style.display = 'none';

    switch (section) {
        case 'upload':
            uploadSection.style.display = 'block';
            break;
        case 'processing':
            processingSection.style.display = 'block';
            break;
        case 'complete':
            completeSection.style.display = 'block';
            break;
        case 'error':
            errorSection.style.display = 'block';
            break;
    }
}

function updateProgress(percentage, status) {
    progressFill.style.width = `${percentage}%`;
    progressText.textContent = `${percentage}%`;
    statusText.textContent = status;
}

function showError(message) {
    errorMessage.textContent = message;
    showSection('error');
}

// Initialize
showSection('upload');
