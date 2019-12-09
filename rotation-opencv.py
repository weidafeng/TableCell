# -*- coding: utf-8 -*-
'''
博客1：python+opencv实现基于傅里叶变换的旋转文本校正
https://blog.csdn.net/qq_36387683/article/details/80530709

博客2：OpenCV—python 图像矫正（基于傅里叶变换—基于透视变换）
https://blog.csdn.net/wsp_1138886114/article/details/83374333


傅里叶相关知识：
https://blog.csdn.net/on2way/article/details/46981825
频率：对于图像来说就是指图像颜色值的梯度，即灰度级的变化速度
幅度：可以简单的理解为是频率的权，即该频率所占的比例
DFT之前的原图像在x y方向上表示空间坐标，DFT是经过x y方向上的傅里叶变换来统计像素在这两个方向上不同频率的分布情况，
所以DFT得到的图像在x y方向上不再表示空间上的长度，而是频率。

仿射变换与透射变换：
仿射变换和透视变换更直观的叫法可以叫做“平面变换”和“空间变换”或者“二维坐标变换”和“三维坐标变换”.
从另一个角度也能说明三维变换和二维变换的意思，仿射变换的方程组有6个未知数，所以要求解就需要找到3组映射点，
三个点刚好确定一个平面.
透视变换的方程组有8个未知数，所以要求解就需要找到4组映射点，四个点就刚好确定了一个三维空间.


图像旋转算法 数学原理：
https://blog.csdn.net/liyuan02/article/details/6750828


角度angle可以用np.angle()
ϕ=atan(实部/虚部)
numpy包中自带一个angle函数可以直接根据复数的实部与虚部求出角度（默认出来的角度是弧度）。
'''

import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

def fourier_demo():
    #1、读取文件，灰度化
    img = cv.imread('img/table-3.png')
    cv.imshow('original', img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)

    #2、图像延扩
    # OpenCV中的DFT采用的是快速算法，这种算法要求图像的尺寸是2的、3和5的倍数是处理速度最快。
    # 所以需要用getOptimalDFTSize()
    # 找到最合适的尺寸，然后用copyMakeBorder()填充多余的部分。
    # 这里是让原图像和扩大的图像左上角对齐。填充的颜色如果是纯色，
    # 对变换结果的影响不会很大，后面寻找倾斜线的过程又会完全忽略这一点影响。
    h, w = img.shape[:2]
    new_h = cv.getOptimalDFTSize(h)
    new_w = cv.getOptimalDFTSize(w)
    right = new_w - w
    bottom = new_h - h
    nimg = cv.copyMakeBorder(gray, 0, bottom, 0, right, borderType=cv.BORDER_CONSTANT, value=0)
    cv.imshow('optim image', nimg)

    #3、执行傅里叶变换，并得到频域图像
    f = np.fft.fft2(nimg) # 将图像从空间域转到频域
    fshift = np.fft.fftshift(f)  # 将低频分量移动到中心，得到复数形式（实部、虚部）
    magnitude = np.log(np.abs(fshift))  # 用abs()得到实数(imag()得到虚部），取对数是为了将数据变换到0-255，相当与实现了归一化。

    # 4、二值化，进行Houge直线检测
    # 二值化
    magnitude_uint = magnitude.astype(np.uint8) #HougnLinesP()函数要求输入图像必须为8位单通道图像
    ret, thresh = cv.threshold(magnitude_uint, thresh=11, maxval=255, type=cv.THRESH_BINARY)
    print("ret:",ret)
    cv.imshow('thresh', thresh)
    print("thresh.dtype:", thresh.dtype)
    #霍夫直线变换
    lines = cv.HoughLinesP(thresh, 2, np.pi/180, 30, minLineLength=40, maxLineGap=100)
    print("len(lines):", len(lines))

    # 5、创建一个新图像，标注直线，找出偏移弧度
    #创建一个新图像，标注直线
    lineimg = np.ones(nimg.shape,dtype=np.uint8)
    lineimg = lineimg * 255

    piThresh = np.pi/180
    pi2 = np.pi/2
    print("piThresh:",piThresh)
    # 得到三个角度，一个是0度，一个是90度，另一个就是我们需要的倾斜角。
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(lineimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if x2 - x1 == 0:
            continue
        else:
            theta = (y2 - y1) / (x2 - x1)
        if abs(theta) < piThresh or abs(theta - pi2) < piThresh:
            continue
        else:
            print("theta:",theta)

    # 6、计算倾斜角，将弧度转换成角度，并注意误差
    angle = math.atan(theta)
    print("angle（弧度）:",angle)
    angle = angle * (180 / np.pi)
    print("angle（角度1）:",angle)
    angle = (angle - 90)/ (w/h)
    #由于DFT的特点，只有输出图像是正方形时，检测到的角才是文本真正旋转的角度。
    # 但是我们的输入图像不一定是正方形的，所以要根据图像的长宽比改变这个角度。
    print("angle（角度2）:",angle)

    # 7、校正图片
    # 先用getRotationMatrix2D()获得一个仿射变换矩阵，再把这个矩阵输入warpAffine()，
    # 做一个单纯的仿射变换,得到校正的结果：
    center = (w//2, h//2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    cv.imshow('line image', lineimg)
    cv.imshow('rotated', rotated)

if __name__ == '__main__':

    fourier_demo()
    cv.waitKey(0)
    cv.destroyAllWindows()
