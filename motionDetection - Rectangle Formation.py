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
    gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)    # It is easier to find contour from grayscale
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _,thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)    # boundRect returns (x,y) coordinates and height and width
        # We will neglect all the elements with area less than some value
        if cv2.contourArea(contour) < 700:
            continue
        # Else a rectangle will be drawn
        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement Detected'), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,100,150),2)
    
    cv2.imshow("Motion Detector", frame1)
    frame1 = frame2
    ret, frame2 = capture.read()
    
    
    if cv2.waitKey(1) == 27:
        break


# All windows are destroyed and resources are released
cv2.destroyAllWindows()
capture.release()
