'''
import cv2 as cv
import os
import time

#cap = cv.VideoCapture("F:/科研/视频和分析_2023.08.09 16-49-09/videoTracker.avi")
cap = cv.VideoCapture("F:/科研/视频_2023.7.24_60_20_25_60/output_video_cut.avi")
#cap = cv.VideoCapture('00191.MTS')
pos = cap.get(cv.CAP_PROP_POS_FRAMES)
x = cap.get(cv.CAP_PROP_POS_FRAMES)
n=0

while True:
    success, frame = cap.read()
    frame = frame[30:1000,730:1700]
    frame = cv.resize(frame, None, fx=0.7, fy=0.7)
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(n), (15, 15),cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    cv.imshow('frame', frame)
    #cv.waitKey(1)
 
    key = cv.waitKey(100000)
    if key ==  ord("a"):
        pos = cap.get(cv.CAP_PROP_POS_FRAMES) + 2
        cap.set(cv.CAP_PROP_POS_FRAMES, pos)
        n=n-1
    elif key == ord("d"):
        pos = cap.get(cv.CAP_PROP_POS_FRAMES) + 0
        cap.set(cv.CAP_PROP_POS_FRAMES,pos)
        n=n+1
    elif key == ord("q"):
        pos = cap.get(cv.CAP_PROP_POS_FRAMES) + 500
        cap.set(cv.CAP_PROP_POS_FRAMES,pos)
        n=n-499
    elif key == ord("f"):
        pos = cap.get(cv.CAP_PROP_POS_FRAMES) + 8
        cap.set(cv.CAP_PROP_POS_FRAMES,pos)
        n=n+9
    elif key == ord("g"):
        pos = cap.get(cv.CAP_PROP_POS_FRAMES) + 20
        cap.set(cv.CAP_PROP_POS_FRAMES,pos)
        n=n+21
    elif key == ord("h"):
        pos = cap.get(cv.CAP_PROP_POS_FRAMES) + 200
        cap.set(cv.CAP_PROP_POS_FRAMES,pos)
        n=n+201

cap.release()
cv.destroyAllWindows()
'''

import cv2 as cv

#cap = cv.VideoCapture("F:/科研/视频_2023.7.24_60_20_25_60/output_video_cut.avi")
#cap = cv.VideoCapture("F:/科研/视频和分析_2023.08.09 16-49-09/videoTracker.avi")
#cap = cv.VideoCapture("F:/科研/视频_2023.8.19_60_20_not_naive/output_video_cut.avi")
#cap = cv.VideoCapture("F:/科研/视频和分析_2023.08.23 21-49-59/videoTracker.avi")
#cap = cv.VideoCapture("F:/科研/视频和分析_2023.10.20 15-41-45/videoTracker.avi")
#cap = cv.VideoCapture("F:/科研/视频_2023.10.13/output_video_cut.avi")
cap = cv.VideoCapture("F:/科研/视频_2023.10.13/output_video_cut_3.avi")

pos = cap.get(cv.CAP_PROP_POS_FRAMES)
n = 0

while True:
    success, frame = cap.read()
    
    if not success:
        break
    
    #frame = frame[0:1015, 685:1700]
    #frame = frame[10:1040,620:1650]
    frame = frame[15:1075,590:1650]
    frame = cv.resize(frame, None, fx=0.8, fy=0.8)
    #frame = cv.resize(frame, None, fx=1.2, fy=1.2)
    
    cv.putText(frame, f"Frame: {pos}", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.imshow('frame', frame)
 
    key = cv.waitKey(100000)
    if key == ord("a"):
        pos -= 1
        cap.set(cv.CAP_PROP_POS_FRAMES, pos)
        n -= 1
    elif key == ord("d"):
        pos += 1
        cap.set(cv.CAP_PROP_POS_FRAMES, pos)
        n += 1
    elif key == ord("q"):
        pos -= 500
        cap.set(cv.CAP_PROP_POS_FRAMES, pos)
        n -= 500
    elif key == ord("f"):
        pos += 8
        cap.set(cv.CAP_PROP_POS_FRAMES, pos)
        n += 8
    elif key == ord("g"):
        pos += 20
        cap.set(cv.CAP_PROP_POS_FRAMES, pos)
        n += 20
    elif key == ord("h"):
        pos = cap.get(cv.CAP_PROP_POS_FRAMES) + 200
        cap.set(cv.CAP_PROP_POS_FRAMES, pos)
        n += 200

cap.release()
cv.destroyAllWindows()