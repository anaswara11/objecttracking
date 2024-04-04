import cv2
import numpy as np
from object_detection import ObjectDetection
import math

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("classvideo3.mov")

# Get screen resolution
screen_width, screen_height = 1366, 768  # Default values, replace with your screen resolution

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects on frame
    class_ids, _, boxes = od.detect(frame)

    # Iterate through detected objects
    for i in range(len(class_ids)):
        class_id = class_ids[i]
        box = boxes[i]

        # Check if the detected object is a person (class ID 0 for COCO dataset)
        if class_id == 0:
            (x, y, w, h) = box

            # Draw green rectangle for person detection
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Calculate scaling factor
    scale_factor = min(screen_width / 3840, screen_height / 2160) * 0.8

    # Resize frame
    frame_resized = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

    cv2.imshow("Frame", frame_resized)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
