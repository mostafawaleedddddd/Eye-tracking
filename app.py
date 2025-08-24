from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from datetime import datetime
import time
import os
import threading
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Global variables for tracking state
tracking_active = False
blink_count = 0
closed_frames = 0
open_frames = 0
blink_in_progress = False
CLOSED_FRAMES_THRESHOLD = 2
OPEN_FRAMES_THRESHOLD = 2

# Performance configuration
CAMERA_FPS = 30          # Camera capture FPS
PROCESSING_DELAY = 0.02  # Delay between frame processing (seconds)
WEB_DISPLAY_FPS = 30     # FPS for web display

# Video capture object
cap = None
tracking_thread = None

def detect_eyes(frame):
    """Detect eyes in the frame and return processed frame with eye count"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(100, 100))
    eyes_detected = 0
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        upper_face = roi_gray[0:h//2, 0:w]
        eyes = eye_cascade.detectMultiScale(upper_face, 1.1, 3)
        
        for (ex, ey, ew, eh) in eyes:
            eyes_detected += 1
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            _, threshold = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                pupil = max(contours, key=cv2.contourArea)
                M = cv2.moments(pupil)
                if M["m00"] != 0:
                    pupil_x = int(M["m10"] / M["m00"]) + ex
                    pupil_y = int(M["m01"] / M["m00"]) + ey
                    cv2.circle(roi_color, (pupil_x, pupil_y), 2, (0, 0, 255), -1)
    
    return frame, eyes_detected

def update_blink_count(eyes_detected):
    """Update blink counting logic"""
    global blink_count, closed_frames, open_frames, blink_in_progress
    
    is_closed = (eyes_detected == 0)
    if is_closed:
        closed_frames += 1
        open_frames = 0
        if closed_frames >= CLOSED_FRAMES_THRESHOLD:
            blink_in_progress = True
    else:
        open_frames += 1
        closed_frames = 0
        if blink_in_progress and open_frames >= OPEN_FRAMES_THRESHOLD:
            blink_count += 1
            blink_in_progress = False

def tracking_worker():
    """Background thread for eye tracking"""
    global tracking_active, cap, blink_count
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    print("Eye tracking started in background thread")
    
    while tracking_active:
        ret, frame = cap.read()
        if not ret:
            continue
            
        processed_frame, eyes_detected = detect_eyes(frame)
        update_blink_count(eyes_detected)
        
        # Add blink count overlay
        cv2.putText(
            processed_frame,
            f'Blinks: {blink_count}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )
        
        # Add status overlay
        cv2.putText(
            processed_frame,
            'TRACKING ACTIVE',
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Small delay to control frame rate - reduced for higher FPS
        time.sleep(0.02)  # Reduced from 0.05 to 0.02 for ~50 FPS processing
    
    if cap:
        cap.release()
    print("Eye tracking stopped")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/start_tracking')
def start_tracking():
    """Start eye tracking"""
    global tracking_active, tracking_thread, blink_count
    
    if not tracking_active:
        tracking_active = True
        blink_count = 0  # Reset blink counter
        tracking_thread = threading.Thread(target=tracking_worker)
        tracking_thread.daemon = True
        tracking_thread.start()
        return jsonify({'status': 'success', 'message': 'Tracking started'})
    else:
        return jsonify({'status': 'error', 'message': 'Tracking already active'})

@app.route('/stop_tracking')
def stop_tracking():
    """Stop eye tracking"""
    global tracking_active, cap
    
    tracking_active = False
    if cap:
        cap.release()
        cap = None
    return jsonify({'status': 'success', 'message': 'Tracking stopped'})

@app.route('/get_status')
def get_status():
    """Get current tracking status and blink count"""
    global tracking_active, blink_count
    return jsonify({
        'tracking_active': tracking_active,
        'blink_count': blink_count
    })

@app.route('/get_frame')
def get_frame():
    """Get current frame from camera"""
    global cap, tracking_active
    
    if not tracking_active or cap is None:
        return jsonify({'error': 'Camera not active'})
    
    ret, frame = cap.read()
    if not ret:
        return jsonify({'error': 'Failed to capture frame'})
    
    # Process frame for eye detection
    processed_frame, eyes_detected = detect_eyes(frame)
    update_blink_count(eyes_detected)
    
    # Add overlays
    cv2.putText(
        processed_frame,
        f'Blinks: {blink_count}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )
    
    # Convert frame to base64 for web display - optimized for higher FPS
    _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])  # Reduced quality for faster encoding
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'frame': frame_base64,
        'blink_count': blink_count,
        'eyes_detected': eyes_detected
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
