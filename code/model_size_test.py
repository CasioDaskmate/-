import cv2 as cv
import numpy as np
import os
import time
from random import randint
from openpyxl import Workbook

cap = cv.VideoCapture("F:/科研/视频_2024.03.03/output_video.avi")
#cap = cv.VideoCapture("F:/科研/视频_2023.9.28/output_video_cut_1.avi")
#cap = cv.VideoCapture("F:/科研/视频_2023.10.13/output_video_cut.avi")
#cap = cv.VideoCapture("F:\科研\视频_2023.8.19_60_20_not_naive\output_video_cut.avi")
success, frame = cap.read()
height, width = frame.shape[:2]
print(height,width)
frame = frame[0:1080,450:1530] 
#frame = frame[15:1075,590:1650]
frame = cv.resize(frame, None, fx=0.7, fy=0.7)
height, width = frame.shape[:2]
print(height,width)
cv.imshow('first_step', frame)
cv.waitKey(0)


