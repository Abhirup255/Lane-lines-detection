import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- CORE FUNCTIONS FROM YOUR PROJECT ---

def HSL_color_selection(image):
    converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # White mask
    lower_white = np.uint8([0, 200, 0])
    upper_white = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted_image, lower_white, upper_white)
    # Yellow mask
    lower_yellow = np.uint8([10, 0, 100])
    upper_yellow = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted_image, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=mask)

def region_selection(image):
    mask = np.zeros_like(image)
    ignore_mask_color = 255 if len(image.shape) == 2 else (255,) * image.shape[2]
    rows, cols = image.shape[:2]
    vertices = np.array([[
        [cols * 0.1, rows * 0.95],
        [cols * 0.4, rows * 0.6],
        [cols * 0.6, rows * 0.6],
        [cols * 0.9, rows * 0.95]
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(image, mask)

def process_frame(image):
    # Pipeline: HSL -> Gray -> Blur -> Canny -> ROI -> Hough
    hsl_select = HSL_color_selection(image)
    gray = cv2.cvtColor(hsl_select, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    region = region_selection(edges)
    
    lines = cv2.HoughLinesP(region, 1, np.pi/180, 20, minLineLength=20, maxLineGap=300)
    
    line_img = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), [0, 255, 0], 10)
    
    return cv2.addWeighted(image, 0.8, line_img, 1.0, 0.0)

# --- STREAMLIT INTERFACE ---

st.set_page_config(page_title="Lane Detection AI", layout="wide")
st.title("🛣️ Autonomous Lane Lines Detection")
st.write("Upload a driving photo to test the OpenCV pipeline.")

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # Read image
    img = Image.open(uploaded_file)
    img_array = np.array(img)
    # Convert RGB (Streamlit) to BGR (OpenCV)
    cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Process
    with st.spinner('Detecting lanes...'):
        result = process_frame(cv_img)
    
    # Show side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(img, use_container_width=True)
    with col2:
        st.subheader("Lane Detection")
        # Convert back to RGB for display
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)