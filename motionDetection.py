# Motion Detection using Python and OpenCV

import cv2
import numpy as np

# video Camera instance is created
capture = cv2.VideoCapture(0)
ret, frame1 = capture.read()
ret, frame2 = capture.read()

# Loop is used for frames
while capture.isOpened():
    diff = cv2.absdiff(frame1, frame2)              # Difference between two frames
    gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)   # It is easier to find contour from grayscale
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _,thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame1, contours, -1, (0,255,0), 2)
    
    cv2.imshow("Motion Detector", frame1)
    frame1 = frame2
    ret, frame2 = capture.read()
    
    
    if cv2.waitKey(1) == 27:
        break


# All windows are destroyed and resources are released
cv2.destroyAllWindows()
capture.release()
