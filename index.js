import { FilesetResolver, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js";

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const warning = document.getElementById('warning');
const status = document.getElementById('status');

// Initialize the MediaPipe Pose Landmarker
const landmarker = await PoseLandmarker.createFromOptions(await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"), {
    baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task`,
        delegate: "GPU"
    },
    runningMode: "VIDEO",
    numPoses: 1
});
status.textContent = "Model loaded.";
// Enable camera access after model is loaded

// Access the rear camera
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    status.textContent = "getUserMedia is not supported by your browser.";
}

navigator.mediaDevices.getUserMedia({
    video: {
        facingMode: 'environment' // Request the rear camera
    }
}).then(stream => {
    status.textContent += " Camera permission ganted.";
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
}).catch(err => {
    status.textContent = `Error accessing camera: ${err}`;
    console.error(err);
});

// Perform pose estimation on the video stream
function predictWebcam() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    let startTimeMs = performance.now();
    landmarker.detectForVideo(video, startTimeMs, detectionResults => {
        ctx.save();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (detectionResults.landmarks.length >= 1) {
            const landmarks = detectionResults.landmarks[0];
            // Draw the landmarks and connections
            drawConnectors(ctx, landmarks, PoseLandmarker.POSE_CONNECTIONS, {
                color: '#00FF00',
                lineWidth: 2
            });
            drawLandmarks(ctx, landmarks, {
                color: '#FF0000',
                lineWidth: 2
            });

            // --- Prone Detection Logic ---
            // Hip landmarks: 23 (left), 24 (right)
            // Shoulder landmarks: 11 (left), 12 (right)

            const leftHip = landmarks[23];
            const rightHip = landmarks[24];
            const leftShoulder = landmarks[11];
            const rightShoulder = landmarks[12];

            if (leftHip && rightHip && leftShoulder && rightShoulder) {
                // Calculate the center of the hips and the shoulders
                const hipCenterX = (leftHip.x + rightHip.x) / 2;
                const hipCenterY = (leftHip.y + rightHip.y) / 2;
                const shoulderCenterX = (leftShoulder.x + rightShoulder.x) / 2;
                const shoulderCenterY = (leftShoulder.y + rightShoulder.y) / 2;

                // Vector from hip center to shoulder center (approximating the spine direction)
                const spineVector = {
                    x: shoulderCenterX - hipCenterX,
                    y: shoulderCenterY - hipCenterY,
                };
                // Vector from right shoulder to left shoulder (approximating shoulder width)
                const shoulderVector = {
                    x: leftShoulder.x - rightShoulder.x,
                    y: leftShoulder.y - rightShoulder.y,
                };

                // Calculate the cross product
                const crossProductZ = (spineVector.x * shoulderVector.y) - (spineVector.y * shoulderVector.x);

                if (crossProductZ < 0) {
                    warning.textContent = "WARNING: Baby is on tummy!";
                    // Add sound warning here if desired
                } else {
                    warning.textContent = ""; // Clear warning if not prone
                }
            } else {
                warning.textContent = "Detecting pose..."; // Indicate detection is in progress
            }
            // --- End Prone Detection Logic ---
        } else {
            warning.textContent = "No pose"; // Indicate no pose was found
        }

        ctx.restore();
    });

    // Call this function again to continue predicting
    // window.requestAnimationFrame(predictWebcam); // fast as possible
    setTimeout(predictWebcam, 500); // not faster than 2Hz
};

// Helper function to draw landmarks (from MediaPipe examples)
function drawLandmarks(ctx, landmarks, style) {
    ctx.fillStyle = style.color;
    for (const landmark of landmarks) {
        ctx.beginPath();
        ctx.arc(landmark.x * ctx.canvas.width, landmark.y * ctx.canvas.height, style.lineWidth, 0, 2 * Math.PI);
        ctx.fill();
    }
}

// Helper function to draw connections (from MediaPipe examples)
function drawConnectors(ctx, landmarks, connections, style) {
    ctx.strokeStyle = style.color;
    ctx.lineWidth = style.lineWidth;
    for (const connection of connections) {
        const start = landmarks[connection.start];
        const end = landmarks[connection.end];
        ctx.beginPath();
        ctx.moveTo(start.x * ctx.canvas.width, start.y * ctx.canvas.height);
        ctx.lineTo(end.x * ctx.canvas.width, end.y * ctx.canvas.height);
        ctx.stroke();
    }
}
