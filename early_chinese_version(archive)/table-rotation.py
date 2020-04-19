# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         table-rotation.py
# Author:       wdf
# Date:         2019/7/7
# IDE：         PyCharm 
# Description:
#     1. HoughLines ——> get the rotation angle
#     2. warpAffine ——> affine(rotation)
# 输入一张倾斜的图像,自动仿射变换、旋转调整整个图像
#     原文：https: // blog.csdn.net / qq_37674858 / article / details / 80708393
# Usage：
#     1. input： raw image
#-------------------------------------------------------------------------------


import math
import cv2
import numpy as np

# 利用hough line 得到最长的直线对应的角度（旋转角度）
# 默认只显示最长的那条直线
def get_rotation_angle(image, show_longest_line=True, show_all_lines=False):
    image = image.copy() # 复制备份，因为line（）函数为in-place
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)  # canny， 便于hough line减少运算量
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=60) # 参数很关键
    # minLineLengh(线的最短长度，比这个短的都被忽略)
    # maxLineCap（两条直线之间的最大间隔，小于此值，认为是一条直线）
    # 函数cv2.HoughLinesP()是一种概率直线检测，原理上讲hough变换是一个耗时耗力的算法，
    # 尤其是每一个点计算，即使经过了canny转换了有的时候点的个数依然是庞大的，
    # 这个时候我们采取一种概率挑选机制，不是所有的点都计算，而是随机的选取一些个点来计算，相当于降采样。
    lengths = []  # 存储所有线的坐标、长度
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = ((x1-x2)**2 + (y1-y2)**2)**0.5  # 勾股定理，求直线长度
        lengths.append([x1, y1, x2, y2, length])
        # print(line, length)
        if show_all_lines:
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 2) # 绘制所有直线（黑色）
    # 绘制最长的直线
    lengths.sort(key=lambda x: x[-1])
    longest_line = lengths[-1]
    print("longest_line: ",longest_line)
    x1, y1, x2, y2, length= longest_line
    if show_longest_line:
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2) # 绘制直线（红色）
        cv2.imshow("longest", image)
    # 计算这条直线的旋转角度
    angle = math.atan((y2-y1)/(x2-x1))
    print("angle-radin:", angle) # 弧度形式
    angle = angle*(180 /math.pi)
    print("angle-degree:",angle) # 角度形式
    return angle


# 输入图像、逆时针旋转的角度，旋转整个图像（解决了旋转后图像缺失的问题）
def rotate_bound(image, angle):
    # 旋转中心点，默认为图像中心点
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # 给定旋转角度后，得到旋转矩阵
    # 数学原理：
    #       https://blog.csdn.net/liyuan02/article/details/6750828
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)  # 得到旋转矩阵，1.0表示与原图大小一致
    # print("RotationMatrix2D：\n", M)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算旋转后的图像大小（避免图像裁剪）
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵（避免图像裁剪）
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    print("RotationMatrix2D：\n", M)

    # 执行仿射变换、得到图像
    return cv2.warpAffine(image, M, (nW, nH),borderMode=cv2.BORDER_REPLICATE)
    # borderMode=cv2.BORDER_REPLICATE 使用边缘值填充
    # 或使用borderValue=(255,255,255)) # 使用常数填充边界（0,0,0）表示黑色

def main():
    img_path = "./img1/IMG_20190723_162452.jpg"
    # img_path = "./img/table-1.png"

    img = cv2.imread(img_path)
    angle = get_rotation_angle(img, show_longest_line=False, show_all_lines=False)

    imag = rotate_bound(img, angle) # 关键
    # cv2.imshow("raw",img)
    # cv2.imshow('rotated', imag)
    cv2.imwrite('rotated.png', imag)

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()