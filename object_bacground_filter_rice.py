import cv2 as cv
import numpy as np
# cap = cv.VideoCapture(r"E:\Study\Self-study\Python_self_learning\Computer vision\OpenCV\videos\rice_package.mp4")
dir = "/mnt/DE42FB9242FB6DA1/Study/Self-study/Python_self_learning/Computer vision/OpenCV/videos/rice_package.mp4"
cap = cv.VideoCapture(dir)

# Create the tool to detect the contours of the objects in video
# backsub = cv.createBackgroundSubtractorMOG2 (history=500, varThreshold=35, detectShadows=True)	
backsub = cv.createBackgroundSubtractorKNN(history=500, dist2Threshold=1800, detectShadows=True)

valid_contours = 0

while True: 
    ret, frame = cap.read()
    height, width, _ = frame.shape
    # Extract the ROI of the object
    roi = frame[60:390, 175:580]
    backsub_vid = backsub.apply(roi) 


    # Discard the noise away by using cv.threshold and count the contours
    threshold, thres = cv.threshold(backsub_vid, 10, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv.contourArea(contour)
        print(area)
        if area > 35000:
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(roi, (x,y), (x+w, y+h), (0,255,0), 3)
             
    cv.drawContours(roi, [contour], -1, (0,0,150), thickness=2)
    cv.rectangle(frame, (20,200), (30,200), (255,255,255))
    cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (0,255), cv.FONT_HERSHEY_TRIPLEX, 1, (0,255,0), thickness=1)
    cv.putText(frame, str(f"Count(s): {round(valid_contours)}"), (0,200), cv.FONT_HERSHEY_TRIPLEX, 1, (0,255,0), thickness=1)
    
    cv.imshow("Bao1", frame)
    cv.imshow("BackSub_bao1", backsub_vid)
    cv.imshow("BackSub_bao1_thres", thres)  
    cv.imshow("ROI", roi)
    
    key = cv.waitKey(20)
    if key == 27: 
        break

cap.release()
cv.destroyAllWindows()

