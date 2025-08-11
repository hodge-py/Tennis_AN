import cv2
import numpy as np

# Video input
cap = cv2.VideoCapture("basketball_video.mp4")

# List to store past positions
points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV for color-based ball detection (example: orange basketball)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Largest contour
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)

        if radius > 5:  # Avoid noise
            center = (int(x), int(y))
            points.append(center)

            # Draw the ball
            cv2.circle(frame, center, int(radius), (0, 255, 255), 2)

    # Draw trailing line
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(frame, points[i - 1], points[i], (0, 0, 255), 2)

    cv2.imshow("Basketball Tracking with Trail", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
