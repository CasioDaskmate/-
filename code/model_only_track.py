import cv2 as cv
import numpy as np
import os
import time
from random import randint
from openpyxl import Workbook

def background(image):
    frame = mask(image)
    #blur = cv.pyrMeanShiftFiltering(image,sp=60,sr=60)
    #d_frame = cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
    #d_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    d_frame = cv.GaussianBlur(frame,(3,3),0)
    ret, binary = cv.threshold(d_frame, 98, 255, cv.THRESH_BINARY_INV)
    ret, binary = cv.threshold(binary, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,1))
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel=kernel, iterations=1)
    sure_bg = cv.dilate(opening, kernel, iterations=1)
    cv.imshow('b',sure_bg)
    return sure_bg

def mask(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, frame = cv.threshold(frame, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    height, width = frame.shape[:2]
    #print(height,width)
    #center_x, center_y = 325,325
    center_x, center_y = 370,370
    r = 340
    mask = np.zeros((height, width), dtype=np.uint8)
    cv.circle(mask, (center_x, center_y), r, (255, 255, 255), -1)
    frame = cv.bitwise_and(frame, frame, mask=mask)
    cv.circle(mask, (center_x, center_y), r, (255, 255, 255), -1)
    ret, frame = cv.threshold(frame, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    return frame




cap = cv.VideoCapture("F:/科研/视频_2023.10.13/output_video_cut.avi")
success, frame = cap.read()
#print(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)))
#print(int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
#frame = frame[0:1010,685:1695]
frame = frame[15:1075,590:1650]
frame = cv.resize(frame, None, fx=0.7, fy=0.7)
d_frame = background(frame)
#d_frame = water(frame)
counter_ids = []
colors = []
ellipses_dict = {}

# 完成输出设置
time_t = time.strftime('%Y.%m.%d %H-%M-%S', time.localtime(time.time()))
outDir = 'F:/科研/视频和分析_' + time_t
os.mkdir(outDir)
outFile_1 = outDir + '\\videoTracker.avi'
r = cap.get(cv.CAP_PROP_FPS)
fourcc=cv.VideoWriter_fourcc(*'XVID')
write = cv.VideoWriter(outFile_1, fourcc, r, (742,742), True)
outFile_2 = outDir + 'NND_' + time_t + '.xlsx'
f=0
pos = cap.get(cv.CAP_PROP_POS_FRAMES)


while True:
    contours = []
    contours_tmp, hierarchy = cv.findContours(d_frame, cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    for i, c in enumerate(contours_tmp):
        if not (cv.arcLength(c,True) > 180 or cv.arcLength(c,True) < 50 or len(c) < 5):
            contours.append(c)

    for i, c in enumerate(contours):   
        if i not in counter_ids:
            x,y,w,h = cv.boundingRect(c)
            #cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 1)
            counter_ids.append(i)
            colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
            ellipse = cv.fitEllipse(c)
            ellipses_dict[i] = ellipse
            cv.ellipse(frame, ellipse, colors[i], 2)
            cv.putText(frame, str(i+1), (int(ellipse[0][0]),int(ellipse[0][1])), cv.FONT_HERSHEY_PLAIN, 1, [0, 0, 255], 2)
    write.write(frame)
    cv.imshow('first_step', frame)
            
    if cv.waitKey(1) == ord(' '):
        break
    elif cv.waitKey(1) == ord('o'):
        success, frame = cap.read()
        #frame = frame[0:1010,685:1695] 
        #frame = frame[10:1040,620:1650] #8.19
        frame = frame[15:1075,590:1650]
        frame = cv.resize(frame, None, fx=0.7, fy=0.7)

        d_frame = background(frame)
        counter_ids = []
        ellipses_dict = {}

while cap.isOpened():   
    contour_ids_o = []
    success, frame = cap.read()
    #if frame == None:
        #break
    #frame = frame[0:1010,685:1695]
    #frame = frame[10:1040,620:1650]
    frame = frame[15:1075,590:1650]
    frame = cv.resize(frame, None, fx=0.7, fy=0.7)
    d_frame = background(frame)
    #d_frame = water(frame)
    contours = []
    contours_tmp, hierarchy = cv.findContours(d_frame, cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    for i, c in enumerate(contours_tmp):
            if not (cv.arcLength(c,True) > 190 or cv.arcLength(c,True) < 60 or len(c) < 5):
                contours.append(c)

    if len(contours) == 20:
        for i, c in enumerate(contours):   
            #if i not in counter_ids:
            #print(i)
            x,y,w,h = cv.boundingRect(c)
            #cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 1)
            #counter_ids.append(i)
            ellipse = cv.fitEllipse(c)
            #ellipse = list(ellipse)
            #cv.ellipse(frame, ellipse, [128,72,155], 2)
            nearest_ellipse = None
            min_distance = float('inf')
            for i, v in ellipses_dict.items():
                if i not in contour_ids_o:
                    distance = ((v[0][0] - ellipse[0][0]) ** 2 + (v[0][1] - ellipse[0][1]) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        nearest_ellipse = i

            if nearest_ellipse != None:      
                ellipses_dict[nearest_ellipse] = ellipse
                contour_ids_o.append(nearest_ellipse)
                cv.ellipse(frame, ellipse, colors[nearest_ellipse], 2)
                cv.putText(frame, str(nearest_ellipse+1), (int(ellipse[0][0]),int(ellipse[0][1])), cv.FONT_HERSHEY_PLAIN, 1, [0, 0, 255], 2)

    write.write(frame)     
    cv.imshow('okframe', frame)
        
    f=f+1
    if f == 50000:
        break

    #cv.waitKey(1)
    if cv.waitKey(1) == ord('q'):
        break
    #print(ellipses_dict)
 
cap.release()
write.release()
cv.destroyAllWindows()