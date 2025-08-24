import cv2 
import numpy as np 
from datetime import datetime 
import time 
import os

def detect_eyes(frame): 
    # Convert frame to grayscale for better detection as we have only 1 channel not RGB 3 channels #reduce computational time 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load pre-trained cascade classifier for face detection
    #Trained on many positive and negative images
    #Efficient for real-time face and eye detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Detect faces 
    #1.3 is the scale factor (how much the image size is reduced at each image scale)
    #5 is the minimum number of neighbors (higher values = less detections but higher quality)
# More robust face detection with confidence
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(100, 100))
    eyes_detected = 0
    # loop for each face in  the picture 
    for (x, y, w, h) in faces:
        # Define region of interest (ROI) for face 
       # Expand eye search region within face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

# Try to detect eyes in upper half of face (more reliable)
        upper_face = roi_gray[0:h//2, 0:w]
        eyes = eye_cascade.detectMultiScale(upper_face, 1.1, 3)
        # Detect eyes within the face region
        #eyes = eye_cascade.detectMultiScale(roi_gray,1.1,3)
        #loop to get eyes from the faces in the picture 

        for (ex, ey, ew, eh) in eyes:
            eyes_detected += 1
            # Draw rectangle around each eye
            #it takes the dimensions for the rectangle to be drawn and it color is 
            # green (0,255,0),2 is for thickness of the line 
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Extract eye region for processing
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            
            # Apply thresholding to isolate the pupil
            _, threshold = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours in the thresholded image
            #inverts the threshold (dark becomes light and vice versa)
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Find the largest contour (assumed to be the pupil)
                pupil = max(contours, key=cv2.contourArea)
                
                # Calculate the center of the pupil
                M = cv2.moments(pupil)
                if M["m00"] != 0:
                    pupil_x = int(M["m10"] / M["m00"]) + ex
                    pupil_y = int(M["m01"] / M["m00"]) + ey
                    
                    # Draw the pupil center 
                    cv2.circle(roi_color, (pupil_x, pupil_y), 2, (0, 0, 255), -1)

    return frame, eyes_detected

def start_eye_tracking(): 
    # Initialize video capture from webcam we used(0) 
    cap = cv2.VideoCapture(0)

    # Set up video recording
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #Create filename with current date/time

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    #20.0 is frames per second, (640,480) is video size

    out = cv2.VideoWriter(f'eye_tracking_{timestamp}.avi', fourcc, 20.0, (640,480))

    print("Eye tracking started. Press 'q' to quit.")

    # Blink counting state (does not change detection logic)
    blink_count = 0
    closed_frames = 0
    open_frames = 0
    blink_in_progress = False
    CLOSED_FRAMES_THRESHOLD = 2  # number of consecutive frames with 0 eyes to consider eyes closed
    OPEN_FRAMES_THRESHOLD = 2    # number of consecutive frames with eyes detected to confirm reopen

    try:
        while True:
            # Capture frame-by-frame
            # Continuous loop
            # Get a frame from the camera
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
                
            #It processes each frame to detect faces, eyes, and pupils
            #It returns the processed frame and the number of eyes detected
            processed_frame, eyes_detected = detect_eyes(frame)

            # Update blink state machine based on eyes_detected
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

            # Overlay blink count on the frame (for display and recording)
            cv2.putText(
                processed_frame,
                f'Blinks: {blink_count}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )
            
            # Record the processed frame
            out.write(processed_frame)
            
            # Display the result
            cv2.imshow('Eye Tracking', processed_frame)
            
            # Break the loop if 'q' is pressed
            # waits for a key event for 1 millisecond
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:# Always run this cleanup code
        # Release camera and video writer
        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_eye_tracking()