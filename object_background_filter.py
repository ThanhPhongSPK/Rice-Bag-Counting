import cv2 as cv
import numpy as np

def get_centroid(contour):
    M = cv.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

cap = cv.VideoCapture("/mnt/DE42FB9242FB6DA1/Study/Self-study/Python_self_learning/Computer vision/OpenCV/rice_bags/videos/0001-0250.mp4")

# Define some parameters for saving to an video
frame_size = tuple(reversed(cap.read()[1].shape[:2]))
fps = cap.get(cv.CAP_PROP_FPS)
forcc = cv.VideoWriter_fourcc(*"XVID")
output_path = 'bao2_out.mp4'
out = cv.VideoWriter(output_path, forcc, fps, frame_size)

# Create the tool to detect the contours of the objects in video
# backsub = cv.createBackgroundSubtractorMOG2 (history=100, varThreshold=35, detectShadows=True)
backsub = cv.createBackgroundSubtractorKNN(history=500, dist2Threshold=1800, detectShadows=True)

objects_count = 0
trackers = []

while True:

    ret, frame = cap.read()
    # height, width, _ = frame.shape

    # Extract the ROI of the object
    # roi = frame[200:700, 125:450] # bao2
    roi = frame[225:725, 100:800]
    backsub_vid = backsub.apply(roi)

    kernel = np.ones((3, 3), np.uint8)
    backsub_vid = cv.morphologyEx(backsub_vid, cv.MORPH_OPEN, kernel)

    # Discard the noise away by using cv.threshold and count the contours
    threshold, thres = cv.threshold(backsub_vid, 253, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    current_tracker = []

    for contour in contours:
        if contour.all():
            area = cv.contourArea(contour)
            print(area)
            if area > 90000:
                # Get center, bdb of the bag and draw
                centroid = get_centroid(contour)
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(roi, (x,y), (x+w, y+h), (0,255,0), 3)
                cv.circle(roi, centroid, 7, (255,255,255), 2)
                if centroid:
                    same_object = False
                    for tracker in trackers:
                        distance = cv.norm(centroid, tracker)
                        if distance < 50:
                            same_object = True
                            current_tracker.append(centroid)
                            break
                    if not same_object:
                        objects_count += 1
                        current_tracker.append(centroid)

    trackers = current_tracker

    cv.drawContours(roi, [contour], -1, (0,0,150), thickness=2)
    cv.rectangle(frame, (10,200), (30,200), (255,255,255))
    cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (0,255), cv.FONT_HERSHEY_TRIPLEX, 1, (0,255,0), thickness=1)
    cv.putText(frame, str(f"Count(s): {objects_count}"), (0,200), cv.FONT_HERSHEY_TRIPLEX, 1, (0,255,0), thickness=1)

    # Save the frame
    out.write(frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    cv.imshow("Bao", frame)
    # cv.imshow("BackSub_bao1", backsub_vid)
    cv.imshow("BackSub_bao1_thres", thres)
    cv.imshow("ROI", roi)
    key = cv.waitKey(0)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
out.release()