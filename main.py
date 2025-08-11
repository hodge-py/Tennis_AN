import cv2
import numpy as np

# Initialize the webcam
# Use a video file by replacing 0 with the file path: e.g., cv2.VideoCapture("tennis_ball.mp4")
cap = cv2.VideoCapture("videoplayback.mp4")

# Define the HSV color range for a tennis ball
# These values can be tuned for better performance
green_lower = (29, 86, 6)
green_upper = (64, 255, 255)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if the video ends or camera fails

    # 1. Pre-processing
    # Resize frame for faster processing and blur to reduce noise
    frame = cv2.resize(frame, (600, 400))
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # 2. Convert to HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 3. Create a mask for the tennis ball color
    mask = cv2.inRange(hsv, green_lower, green_upper)
    # Refine the mask to remove small imperfections
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 4. Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = None

    # Proceed only if at least one contour was found
    if len(contours) > 0:
        # Find the largest contour in the mask
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Proceed only if the radius meets a minimum size
        if radius > 10:
            # Draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

            # Print the center coordinates
            center = (int(x), int(y))
            # print(f"Ball detected at: {center}")

    # Display the resulting frame
    cv2.imshow("Tennis Ball Tracker", frame)
    # Optional: Display the mask for debugging
    # cv2.imshow("Mask", mask)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()