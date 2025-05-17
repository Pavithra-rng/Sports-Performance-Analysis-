import cv2
import numpy as np
import time

# Load video
video_path = 'video.mp4'  # Replace with your video file
cap = cv2.VideoCapture(video_path)

# Frame rate (needed for time calculation)
fps = cap.get(cv2.CAP_PROP_FPS)

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Variables to calculate speed
prev_center = None
prev_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 450))
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 800:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

            # Current frame time in seconds
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            speed = 0  # default
            if prev_center is not None and prev_time is not None:
                dx = cx - prev_center[0]
                dy = cy - prev_center[1]
                distance = np.sqrt(dx**2 + dy**2)
                dt = current_time - prev_time
                if dt > 0:
                    speed = distance / dt  # pixels per second

            # Update previous data
            prev_center = (cx, cy)
            prev_time = current_time

            # Draw and display speed
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
            cv2.putText(frame, f'Speed: {speed:.2f} px/s', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            break  # track only first major moving object

    cv2.imshow("Player Speed Tracking", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
