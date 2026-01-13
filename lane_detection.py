import cv2
import numpy as np

def select_white_yellow(image):
    hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(hsl, (0, 200, 0), (255, 255, 255))
    yellow_mask = cv2.inRange(hsl, (10, 0, 100), (40, 255, 255))
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=mask)

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)

def region_of_interest(image):
    height, width = image.shape[:2]
    polygon = np.array([[
        (int(0.1 * width), height),
        (int(0.4 * width), int(0.6 * height)),
        (int(0.6 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ]], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image, mask)

def detect_lines(image):
    return cv2.HoughLinesP(image, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=300)

def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return cv2.addWeighted(image, 0.8, line_image, 1, 1)

def process_image(image):
    filtered = select_white_yellow(image)
    edges = detect_edges(filtered)
    roi = region_of_interest(edges)
    lines = detect_lines(roi)
    return draw_lines(image, lines)

# 🎥 Video path (confirmed from your folder)
video_path = r"E:\Downloads\Lane-lines-detection-using-Python-and-OpenCV-master\Lane-lines-detection-using-Python-and-OpenCV-master\test_videos\solidWhiteRight.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    output = process_image(frame)
    cv2.imshow("Lane Detection - Video", output)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
