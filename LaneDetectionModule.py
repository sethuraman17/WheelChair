import cv2
import numpy as np
import utlis

curveList = []
avgVal = 10

def getLaneCurve(img, display=2):

    imgCopy = img.copy()
    imgResult = img.copy()

    imgThresh = utlis.ColorDetection(img, lower_white=np.array([80, 0, 0]), upper_white=np.array([160, 160, 255]))

    hT, wT, c = img.shape
    points = utlis.valTrackbars()
    imgWarp = utlis.warpImg(imgThresh, points, wT, hT)
    imgWarpPoints = utlis.drawPoints(imgCopy, points)

    MiddlePoint, imgHist = utlis.Histogram(imgWarp, display=True, minPer=0.5, region=4)
    CurveAveragePoint, imgHist = utlis.Histogram(imgWarp, display=True, minPer=0.9, region=1)
    curveRaw = MiddlePoint-CurveAveragePoint
    # print(curveRaw)

    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))

    if display != 0:
        imgInvWarp = utlis.warpImg(imgWarp, points, wT, hT, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        midY = 450
        cv2.putText(imgResult, str(curve), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
        cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
        if curve > 0:
            cv2.line(imgResult, (300, 60), (465, 60), (255, 0, 255), 2)
        elif curve < 0:
            cv2.line(imgResult, (160, 175), (300, 175), (255, 0, 255), 2)
        for x in range(-30, 30):
            w = wT // 20
            cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                     (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
        # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        # cv2.putText(imgResult, 'FPS ' + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 50, 50), 3);
    if display == 2:
        imgStacked = utlis.stackImages(0.7, ([img, imgWarpPoints, imgWarp],
                                             [imgHist, imgLaneColor, imgResult]))
        cv2.imshow('ImageStack', imgStacked)
    elif display == 1:
        cv2.imshow('Resutlt', imgResult)

    # cv2.imshow("Image", imgThresh)
    # cv2.imshow("WarpImage", imgWarp)
    # cv2.imshow("WarpImagePoints", imgWarpPoints)
    # cv2.imshow("HistogramImage", imgHist)

    return curve



if __name__ == '__main__':
    cap = cv2.VideoCapture('test1.mp4')
    intialTrackbarVals = [102, 80, 20, 164]
    utlis.initializeTrackbars(intialTrackbarVals)
    frame_count = 0
    curveList = []

    while True:
        frame_count += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
        success, img = cap.read()
        img = cv2.resize(img, (480, 240))
        curve = getLaneCurve(img, display=2)
        print(curve)
        cv2.imshow("Images", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
