import cv2
import numpy as np
from collections import deque

# for keeping center points of object
buffer_size = 16
pts = deque(maxlen=buffer_size)

# blue HSV
blueLower = (84,  98,  0)
blueUpper = (179, 255, 255)

#capture
cap = cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)

while True:

    success, imgOriginal = cap.read()

    if success:

        #blur
        blurred = cv2.GaussianBlur(imgOriginal, (11,11), 0)

        # HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV IMAGE", hsv)
        
        # mask for blue
        mask = cv2.inRange(hsv, blueLower, blueUpper)

        # deleting noises which are in area of mask
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow("Mask + Erosion + Dilation", mask)

        # contours
        contours,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(contours) > 0:

            # get max contour
            c = max(contours, key=cv2.contourArea)

            # return rectangle
            rect = cv2.minAreaRect(c)
            ((x,y), (width, height), rotation) = rect

            s = f"x {np.round(x)}, y: {np.round(y)}, width: {np.round(width)}, height: {np.round(height)}, rotation: {np.round(rotation)}"
            print(s)

            # box
            box = cv2.boxPoints(rect)
            box = np.int64(box)

            # moment
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # draw contour
            cv2.drawContours(imgOriginal, [box], 0, (0, 255, 255), 2)

            # point in center
            cv2.circle(imgOriginal, center, 5, (255, 0, 255), -1)

            # print inform
            cv2.putText(imgOriginal, s, (25, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)


        # deque
        pts.appendleft(center)
        for i in range(1, len(pts)):

            if pts[i - 1] is None or pts[i] is None: continue

            cv2.line(imgOriginal, pts[i - 1], pts[i], (0, 255, 0), 3)

        cv2.imshow("DETECTED IMAGE", imgOriginal)



    if cv2.waitKey(1) & 0xFF == ord("q"): break
