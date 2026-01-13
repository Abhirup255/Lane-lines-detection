import streamlit as st
import cv2
import numpy as np
import os
import time  # <--- Essential for smooth playback

# --- CORE LOGIC (Fixed cv2.line) ---
def process_frame(frame):
    hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(hsl, (0, 200, 0), (255, 255, 255))
    yellow_mask = cv2.inRange(hsl, (10, 0, 100), (40, 255, 255))
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    filtered = cv2.bitwise_and(frame, frame, mask=mask)
    
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    height, width = frame.shape[:2]
    polygon = np.array([[(int(0.1*width), height), (int(0.4*width), int(0.6*height)), 
                         (int(0.6*width), int(0.6*height)), (int(0.9*width), height)]], np.int32)
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, polygon, 255)
    roi = cv2.bitwise_and(edges, roi_mask)
    
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 20, minLineLength=20, maxLineGap=300)
    line_image = np.zeros_like(frame)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Ensure all 4 coordinates are present
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
            
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

# --- STREAMLIT UI ---
st.set_page_config(page_title="Auto Lane Detection", layout="centered")
st.title("🎥 Automated Lane Detection Stream")

# Path must match your GitHub folder structure
video_path = os.path.join("test_videos", "solidWhiteRight.mp4")

if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    st_frame = st.empty()  # This is the "screen" that refreshes

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Restart video loop automatically
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        processed = process_frame(frame)
        # Convert BGR to RGB so colors look correct on the web
        st_frame.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB")
        
        # SLOW DOWN the loop slightly so the browser doesn't freeze
        time.sleep(0.03) 
        
    cap.release()
else:
    st.error(f"Cannot find video at: {video_path}")
