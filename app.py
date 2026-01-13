import streamlit as st
import cv2
import numpy as np
import os
import time  # Essential for syncing the video speed on the web

# --- CORE PROCESSING LOGIC ---
def process_frame(frame):
    # 1. Filter for White/Yellow lines (HSL)
    hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(hsl, (0, 200, 0), (255, 255, 255))
    yellow_mask = cv2.inRange(hsl, (10, 0, 100), (40, 255, 255))
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    filtered = cv2.bitwise_and(frame, frame, mask=mask)
    
    # 2. Detect Edges and ROI
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    height, width = frame.shape[:2]
    polygon = np.array([[(int(0.1*width), height), (int(0.4*width), int(0.6*height)), 
                         (int(0.6*width), int(0.6*height)), (int(0.9*width), height)]], np.int32)
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, polygon, 255)
    roi = cv2.bitwise_and(edges, roi_mask)
    
    # 3. Detect and Draw Lines
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 20, minLineLength=20, maxLineGap=300)
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
            
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

# --- STREAMLIT AUTOMATED INTERFACE ---
st.set_page_config(page_title="Live Lane Detection", layout="centered")
st.title("🎥 Real-Time Lane Detection Stream")
st.info("The model below is processing the video frame-by-frame live on the web.")

# Ensure this path matches your GitHub: test_videos/solidWhiteRight.mp4
video_path = os.path.join("test_videos", "solidWhiteRight.mp4")

if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # This creates a persistent "screen" on the web page
    st_frame = st.empty() 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # LOOP: Restarts the video when it ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Process the frame
        processed = process_frame(frame)
        
        # WE MUST CONVERT TO RGB: Streamlit displays RGB, OpenCV uses BGR
        output_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # Display the frame in the placeholder
        st_frame.image(output_rgb, channels="RGB", use_container_width=True)
        
        # SMALL DELAY: Prevents the browser from freezing
        time.sleep(0.01) 
        
    cap.release()
else:
    st.error(f"Cannot find video at: {video_path}")
