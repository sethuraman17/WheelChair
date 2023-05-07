import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture('hostel2.mp4')


# Loop through the frames of the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Convert the frame to grayscale
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_yellow = np.array([18, 94, 140])
    up_yellow = np.array([48, 255, 255])
    mask = cv2.inRange(hsv, low_yellow, up_yellow)

    # Apply edge detection to extract the edges
    edges = cv2.Canny(mask, 50, 150)

    # Apply HoughLinesP to detect lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    # Extract the x, y coordinates of the endpoints of the detected lines
    points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        points.append((x1, y1))
        points.append((x2, y2))

    # Convert the points to numpy array format
    points = np.array(points)

    # Define the RANSAC parameters
    num_iterations = 100
    distance_threshold = 50
    inlier_threshold = 0.7

    # Implement RANSAC to fit a line to the points
    best_line = None
    best_inliers = []
    for i in range(num_iterations):
        # Randomly select two points from the points array
        sample = np.random.choice(points.shape[0], 2, replace=False)
        p1 = points[sample[0]]
        p2 = points[sample[1]]

        # Calculate the parameters of the line connecting the two points
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        c = p2[0]*p1[1] - p1[0]*p2[1]

        # Calculate the distance between each point and the line
        distances = np.abs(a*points[:,0] + b*points[:,1] + c) / np.sqrt(a**2 + b**2)

        # Count the number of inliers (points that are within the distance threshold of the line)
        inliers = distances < distance_threshold

        # If the number of inliers is greater than the inlier threshold, update the best line
        if np.sum(inliers) > inlier_threshold*points.shape[0]:
            if best_line is None or np.sum(inliers) > np.sum(best_inliers):
                best_line = (a, b, c)
                best_inliers = inliers

    # Draw the Hough lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw the RANSAC line
    if best_line is not None:
        x1 = int((-best_line[1] * frame.shape[0] - best_line[2]) / best_line[0])
        y1 = 0
        x2 = int((-best_line[1] * frame.shape[0] + frame.shape[1] - best_line[2]) / best_line[0])
        y2 = frame.shape[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Result', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
