# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         切割每个单元格.py
# Author:       wdf
# Date:         2019/7/7
# IDE：         PyCharm 
# Description:
# python提取图片中的表格内容, 针对水平处理后的表格，可以切割每个单元格
# https://blog.csdn.net/weixin_42194239/article/details/91975662
# 1、图像灰度化处理
# 2、图像二值化处理
# 3、图像腐蚀处理（若得到的横纵交线不清楚，添加膨胀处理）
# 4、获取表格交点坐标
# 5、根据交点集获取单元格轮廓并进行过滤
# Usage：
#-------------------------------------------------------------------------------

from PIL import Image, ImageOps
import cv2
import numpy as np

def split_rec(arr):
    """
    切分单元格
    :param arr:
    :return:
    """
    # 数组进行排序
    arr.sort(key=lambda x: x[0],reverse=True)
    # 数组反转
    arr.reverse()
    for i in range(len(arr) -1 ):
        if arr[i+1][0] == arr[i][0]:
            arr[i+1][3] = arr[i][1]
            arr[i + 1][2] = arr[i][2]
        if arr[i+1][0] > arr[i][0]:
            arr[i + 1][2] = arr[i][0]
        print(arr[i])

    return arr


def get_points(img_transverse, img_vertical):
    """
    获取横纵线的交点
    :param img_transverse:
    :param img_vertical:
    :return:
    """
    img = cv2.bitwise_and(img_transverse, img_vertical)
    return img


def dilate_img(img, kernal_args:tuple, iterations:int):
    """
    dilate image
    @param kernel_args 卷积核参数（2，2）
    @param interations dilate的迭代次数
    """
    kernel = np.ones(kernal_args, np.uint8)
    return cv2.dilate(img, kernel,iterations=iterations)
    pass


def erode_img(img,kernel_args=(2,2),iterations=1):
    """
    对图像进行腐蚀
    @param kernel_args 卷积核参数（2，2）
    @param interations erode的迭代次数
    """
    kernel = np.ones(kernel_args, np.uint8)
    return cv2.erode(img, kernel,iterations=iterations)


def bin_img(img:'numpy.ndarray'):
    """
    对图像进行二值化处理
    :param img: 传入的图像对象（numpy.ndarray类型）
    :return: 二值化后的图像
    """
    ret,binImage=cv2.threshold(img,180,255,cv2.THRESH_BINARY_INV)
    return binImage

def gray_img(img:'numpy.ndarray'):
    """
    对读取的图像进行灰度化处理
    :param img: 通过cv2.imread(imgPath)读取的图像数组对象
    :return: 灰度化的图像
    """
    grayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return grayImage
    pass

def get_rec(img):
    """
    获取单元格
    :param img:
    :return:
    """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [0] * len(contours)
    boundRect = [0] * len(contours)
    rois = []
    for i in range(len(contours) ):
        cnt = contours[i]
        contours_poly[i] = cv2.approxPolyDP(cnt, 1, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        rois.append(np.array(boundRect[i]))
        # img = cv2.rectangle(img_bak, (boundRect[i][0], boundRect[i][1]), (boundRect[i][2], boundRect[i][3]),
        #                     (255, 255, 255), 1, 8, 0)
    rois = split_rec(rois)
    return rois

if __name__ == "__main__":
    image  = "./img/1.jpg"
    img_bak = cv2.imread(image)
    img = gray_img(img_bak)
    img = bin_img(img)
    img_transverse = erode_img(img,(1,2),40)
    img_vertical = erode_img(img, (2,1), 40)
    # img = img_transverse + img_vertical
    img_transverse = dilate_img(img_transverse,(2,2),1)
    img_vertical = dilate_img(img_vertical,(2,2),1)
    img = get_points(img_transverse,img_vertical)

    rois = get_rec(img)
    for i, r in enumerate(rois):
        cv2.imshow("src" + str(i), img_bak[r[3]:r[1], r[2]:r[0]])
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    pass

