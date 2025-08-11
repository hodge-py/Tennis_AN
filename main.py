import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (pretrained, you can replace 'yolov8s.pt' with your custom weights)
model = YOLO('yolov8s.pt')

# Print class names to confirm tennis ball class index
print("Model class names:", model.names)
tennis_ball_class_id = 32  # tennis ball class as you mentioned

# Homography source points (pixels) and destination points (meters)
pts_src = np.array([
    [322, 572],  # bottom left corner in pixels
    [961, 572],  # bottom right corner in pixels
    [900, 418],  # top right corner in pixels
    [379, 419],  # top left corner in pixels
], dtype=np.float32)

pts_dst = np.array([
    [0, 0],           # bottom left corner in meters
    [8.23, 0],        # bottom right corner in meters
    [8.23, 23.77],    # top right corner in meters
    [0, 23.77],       # top left corner in meters
], dtype=np.float32)

# Calculate homography matrix
H, status = cv2.findHomography(pts_src, pts_dst)

def pixel_to_world(px, py, H):
    pts = np.array([[[px, py]]], dtype=np.float32)
    world_pt = cv2.perspectiveTransform(pts, H)
    return world_pt[0][0]  # (X, Y) in meters

# Open video file or webcam
cap = cv2.VideoCapture('Tennis - Made with Clipchamp.mp4')

prev_world_pos = None
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on frame
    results = model(frame)[0]

    tennis_balls = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == tennis_ball_class_id and conf > 0.3:
            tennis_balls.append(box)

    if tennis_balls:
        # Choose highest confidence detection
        best_box = max(tennis_balls, key=lambda b: float(b.conf[0]))
        xyxy = best_box.xyxy[0].cpu().numpy()
        xmin, ymin, xmax, ymax = map(int, xyxy)

        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        # Convert pixel to real world coords
        world_pos = pixel_to_world(center_x, center_y, H)

        # Draw bounding box and center point
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
        cv2.circle(frame, (int(center_x), int(center_y)), 5, (0,0,255), -1)
        cv2.putText(frame, f"Tennis Ball: {conf:.2f}", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Optional: calculate speed if previous position exists
        if prev_world_pos is not None:
            dx = world_pos[0] - prev_world_pos[0]
            dy = world_pos[1] - prev_world_pos[1]
            dist = np.sqrt(dx*dx + dy*dy)
            time_sec = 1 / fps
            speed_m_s = dist / time_sec
            speed_kmh = speed_m_s * 3.6

            cv2.putText(frame, f"Speed: {speed_kmh:.2f} km/h", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        prev_world_pos = world_pos

    cv2.imshow('Tennis Ball Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
