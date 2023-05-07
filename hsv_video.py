import cv2
import numpy as np

def empty(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L-H", "Trackbars", 0, 179, empty)
cv2.createTrackbar("L-S", "Trackbars", 0, 255, empty)
cv2.createTrackbar("L-V", "Trackbars", 0, 255, empty)
cv2.createTrackbar("U-H", "Trackbars", 179, 179, empty)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, empty)
cv2.createTrackbar("U-V", "Trackbars", 255, 255, empty)


while True:
    success, img = cap.read()
    # if not success:
    #     img = cv2.VideoCapture("hostel_test_basic.mp4")
    #     continue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    h_h = cv2.getTrackbarPos("U-H", "Trackbars")
    h_s = cv2.getTrackbarPos("U-S", "Trackbars")
    h_v = cv2.getTrackbarPos("U-V", "Trackbars")

    low_range = np.array([l_h, l_s, l_v])
    up_range = np.array([h_h, h_s, h_v])

    mask = cv2.inRange(hsv, low_range, up_range)

    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Images", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)
    cv2.waitKey(1)
