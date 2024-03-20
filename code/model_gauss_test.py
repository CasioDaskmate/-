import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import time
from random import randint
from openpyxl import Workbook
from scipy.optimize import curve_fit

def background(image):
    frame = mask(image)
    #blur = cv.pyrMeanShiftFiltering(image,sp=60,sr=60)
    #d_frame = cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
    #d_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    d_frame = cv.GaussianBlur(frame,(3,3),0)
    ret, binary = cv.threshold(d_frame, 98, 255, cv.THRESH_BINARY_INV)
    ret, binary = cv.threshold(binary, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel=kernel, iterations=2)
    sure_bg = cv.dilate(opening, kernel, iterations=6)
    cv.imshow('b',sure_bg)
    return sure_bg

def mask(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, frame = cv.threshold(frame, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    height, width = frame.shape[:2]
    print(height,width)
    #center_x, center_y = 325,325
    #r = 320
    center_x, center_y = 617,617
    r = 600
    mask = np.zeros((height, width), dtype=np.uint8)
    cv.circle(mask, (center_x, center_y), r, (255, 255, 255), -1)
    frame = cv.bitwise_and(frame, frame, mask=mask)
    cv.circle(mask, (center_x, center_y), r, (255, 255, 255), -1)
    ret, frame = cv.threshold(frame, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    return frame

# 定义二维高斯分布函数
def gaussian(xy, amplitude, center_x, center_y, sigma_x, sigma_y):
    x, y = xy[:, 0], xy[:, 1]
    return amplitude * np.exp(-((x - center_x) ** 2) / (2 * sigma_x ** 2) - ((y - center_y) ** 2) / (2 * sigma_y ** 2))

cap = cv.imread("D:\\desktop\\research\\code\\3_26440.jpg")
frame = cv.resize(cap, None, fx=1, fy=1)
d_frame = background(frame)
contours = []
counter_ids = []
colors = []
ellipses_dict = {}
datas = np.zeros((1,20))
points = np.zeros((20, 2))
contours_tmp, hierarchy = cv.findContours(d_frame, cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
for i, c in enumerate(contours_tmp):
        if not (cv.arcLength(c,True) > 190 or cv.arcLength(c,True) < 40 or len(c) < 5):
            contours.append(c)

for i, c in enumerate(contours):   
        if i not in counter_ids:
            x,y,w,h = cv.boundingRect(c)
            counter_ids.append(i)
            colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
            ellipse = cv.fitEllipse(c)
            ellipses_dict[i] = ellipse
            points[i][0] =  ellipses_dict[i][0][0]
            points[i][1] =  ellipses_dict[i][0][1]
            #cv.ellipse(frame, ellipse, colors[i], 2)
            #cv.putText(frame, str(i+1), (int(ellipse[0][0]),int(ellipse[0][1])), cv.FONT_HERSHEY_PLAIN, 1, [0, 0, 255], 2)

x_avg = np.mean(points[:, 0])
y_avg = np.mean(points[:, 1])
x_std = np.std(points[:, 0])
y_std = np.std(points[:, 1])
initial_params = [1.0, x_avg, y_avg, x_std, y_std]
#initial_params = [1.0, 600.0, 700.0, 1200.0, 1200.0]
xy_data = np.array([points[:, 0], points[:, 1]]).T
popt, _ = curve_fit(gaussian, xy_data, np.zeros(len(points)), p0=initial_params)
gaussian_center_x, gaussian_center_y = popt[1], popt[2]
print(gaussian_center_x, gaussian_center_y)
#cv.circle(frame, (int(gaussian_center_x), int(gaussian_center_y)), 5, (0, 0, 255), -1)

for i,v in ellipses_dict.items():
    dis = (((v[0][0] - gaussian_center_x) ** 2 + (v[0][1] - gaussian_center_y) ** 2) ** 0.5)
    datas[0][i] = dis
print(datas)

plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
x_range = np.linspace(0, 1, 100)
y_range = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x_range, y_range)
#plt.scatter(points[:, 0], points[:, 1], label='drosophila')
plt.scatter(gaussian_center_x, gaussian_center_y, color='r', marker='d',label = 'center')
circle_radii = np.linspace(0, 160, 4) 
for r in circle_radii:
    circle = plt.Circle((gaussian_center_x, gaussian_center_y), r, color='r', fill=False)
    plt.gca().add_patch(circle)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()