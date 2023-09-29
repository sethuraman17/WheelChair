import cv2
import numpy as np

def ColorDetection(img,lower_white, upper_white):

    imgThresh = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower_white = np.array([80, 0, 0])
    # upper_white = np.array([160, 160, 160])
    maskImg = cv2.inRange(imgThresh, lower_white, upper_white)

    return maskImg

def warpImg(img, points, w, h, inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))

    return imgWarp

def empty(a):
    pass

def initializeTrackbars(intialTrackbarVals, wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("W-T", "Trackbars", intialTrackbarVals[0], wT//2, empty)
    cv2.createTrackbar("W-B", "Trackbars", intialTrackbarVals[2], wT//2, empty)
    cv2.createTrackbar("H-T", "Trackbars", intialTrackbarVals[1], hT, empty)
    cv2.createTrackbar("H-B", "Trackbars", intialTrackbarVals[3], hT, empty)

def valTrackbars(wT=480, hT=240):
    WT = cv2.getTrackbarPos("W-T", "Trackbars")
    WB = cv2.getTrackbarPos("W-B", "Trackbars")
    HT = cv2.getTrackbarPos("H-T", "Trackbars")
    HB = cv2.getTrackbarPos("H-B", "Trackbars")
    points = np.float32([(WT, HT), (wT-WT, HT), (WB, HB), (wT-WB, HB)])

    return points

def drawPoints(img, points):
    for x in range(4):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 15, (0, 0, 255), cv2.FILLED)
    return img

def Histogram(img, minPer=0.5, display=False, region=1):
    if region == 1:
        histValues = np.sum(img, axis=0)
    else:
        histValues = np.sum(img[img.shape[0]//region:, :], axis=0)
    # print(histValues)
    maxValue = np.max(histValues)
    # print(maxValue)
    minVal = minPer*maxValue

    indexArray = np.where(histValues >= minVal)
    basePoint = int(np.average(indexArray))
    # print(basePoint)

    if display:
        imgHist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(histValues):
            cv2.line(imgHist, (x, img.shape[0]), (x, img.shape[0]-intensity//255//region), (255, 0, 255), 1)
            cv2.circle(imgHist, (basePoint, img.shape[0]), 20, (0, 255, 255), cv2.FILLED)

        return basePoint, imgHist

    return basePoint

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

