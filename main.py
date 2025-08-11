import cv2
import numpy as np

frame = cv2.imread("court_frame.jpg")

# Points in the video frame (manually measured)
src_pts = np.float32([
    [320, 450],  # left baseline corner
    [960, 450],  # right baseline corner
    [500, 300],  # left service line corner
    [780, 300]   # right service line corner
])

# Points in the real-world top-down view (choose scale)
dst_pts = np.float32([
    [0, 0],
    [8.23, 0],
    [0, 6.40],
    [8.23, 6.40]
]) * 100  # convert meters to cm for easier scaling

# Compute homography
H, _ = cv2.findHomography(src_pts, dst_pts)

# Warp to bird’s-eye
warped = cv2.warpPerspective(frame, H, (1000, 800))
cv2.imshow("Top-down view", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()