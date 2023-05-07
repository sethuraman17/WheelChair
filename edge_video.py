import cv2
import numpy as np

def make_points(image, line):
    slope, intercept = line
    y1 = int(img.shape[0])
    y2 = int(y1*3.0/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                 left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines


cap = cv2.VideoCapture('test1.mp4')
# cap.set(3, 680)
# cap.set(4, 420)

while True:
    success, img = cap.read()
    # if not success:
    #     video = cv2.VideoCapture("test1.mp4")
    #     continue
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gaussian_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    canny_img = cv2.Canny(gaussian_img, 100, 200)

    mask = np.zeros_like(canny_img)

    height = img.shape[0]       # 420
    # width = img.shape[1]        # 680
    roi_img = np.array([[(200, height),
    (800, 350),
    (1200, height)]], np.int32)
    cv2.fillPoly(mask, roi_img, (255, 255, 255))

    masked_img = cv2.bitwise_and(canny_img, mask)

    line_img = cv2.HoughLinesP(masked_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    averaged_lines = average_slope_intercept(img, line_img)

    display_img = np.zeros_like(img)
    if averaged_lines is not None:
        for line in averaged_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(display_img, (x1, y1), (x2, y2), (0, 0, 255), 10)

    add_weight_img = cv2.addWeighted(img, 0.8, display_img, 1, 1)

    cv2.imshow("Images", img)
    # cv2.imshow("GrayImage", gray_img)
    # cv2.imshow("BlurImage", gaussian_img)
    cv2.imshow("CannyImage", canny_img)
    # cv2.imshow("MaskedImg", masked_img)
    cv2.imshow("AddWeightImage", add_weight_img)

    cv2.waitKey(1)
