from ultralytics import YOLO
import cv2

model = YOLO('yolov8s.pt')

frame = cv2.imread('vlcsnap-2025-08-10-22h54m04s921.png')
results = model(frame)[0]

results.show()  # Visualize detections
print(results)  # Print results to console