# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         hough_line.py
# Author:       wdf
# Date:         2019/7/7
# IDE：         PyCharm 
# Description:  
# Usage：
#-------------------------------------------------------------------------------

import cv2 as cv
import numpy as np
import math
#-----------------霍夫变换---------------------
#前提条件： 边缘检测完成
#标准霍夫线变换
#标准霍夫线变换


# 获取列表的最后一个元素
def takeEnd(elem):
    return elem[-1]

def line_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  #apertureSize参数默认其实就是3
    # cv.imshow("edges", edges)
    lines = cv.HoughLines(edges, 1, np.pi/180, 80)
    for line in lines:
        rho, theta = line[0]  #line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
        a = np.cos(theta)   #theta是弧度
        b = np.sin(theta)
        x0 = a * rho    #代表x = r * cos（theta）
        y0 = b * rho    #代表y = r * sin（theta）
        x1 = int(x0 + 1000 * (-b)) #计算直线起点横坐标
        y1 = int(y0 + 1000 * a)    #计算起始起点纵坐标
        x2 = int(x0 - 1000 * (-b)) #计算直线终点横坐标
        y2 = int(y0 - 1000 * a)    #计算直线终点纵坐标
        # 注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)    #点的坐标必须是元组，不能是列表。
    # cv.imshow("image-lines", image)

def line_detect_possible_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=60)
    # minLineLengh(线的最短长度，比这个短的都被忽略)
    # maxLineCap（两条直线之间的最大间隔，小于此值，认为是一条直线）
    print(len(lines))
    lengths = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = ((x1-x2)**2 + (y1-y2)**2)**0.5
        lengths.append([x1, y1, x2, y2, length])
        # print(line, length)
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2) # 绘制所有直线
    # 绘制最长的直线
    lengths.sort(key=takeEnd)
    longest_line = lengths[-1]
    print(longest_line)
    x1, y1, x2, y2, length= longest_line
    cv.line(image, (x1, y1), (x2, y2), (0, 0, 0), 2) # 绘制直线

    # 计算这条直线的旋转角度
    angle = math.acos((x2-x1)/length)
    print(angle) # 弧度形式
    angle = angle*(180 /math.pi)
    print(angle) # 角度形式

    cv.imshow("longest", image)
    print(lengths)


def main():
    img = cv.imread("./img/rot-45.png")
    # cv.namedWindow("Show", cv.WINDOW_AUTOSIZE)
    # cv.imshow("Show", img)
    line_detect_possible_demo(img)
    # line_detection(img)

    cv.waitKey(0)
    cv.destroyAllWindows()
if __name__ == '__main__':
    main()
