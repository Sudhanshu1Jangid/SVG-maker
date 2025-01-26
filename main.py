import cv2
import numpy as np
import ctypes
import os
import subprocess
from datetime import datetime
from cairosvg import svg2png

def capture_image():
    cap = cv2.VideoCapture('http://your_camera_ip/video')  # Replace with your camera URL
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        cv2.imshow('Press SPACE to capture', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame
