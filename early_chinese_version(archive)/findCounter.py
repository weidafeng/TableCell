# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         findCounter.py
# Author:       wdf
# Date:         2019/7/17
# IDE：         PyCharm 
# Parameters:
#     @param:
#     @param:
# Return： 
#       
# Description:  
# Usage：
#-------------------------------------------------------------------------------
import cv2
import numpy as np

def split_rec(arr):
    """
    切分单元格
    :param arr:
    :return:
    """
    # 数组进行排序
    print(arr)
    print("*"*50)
    arr.sort(key=lambda x: x[0],reverse=True)
    # 数组反转
    arr.reverse()
    for i in range(len(arr) - 1):
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

def get_vertical_line(binary):
    rows, cols = binary.shape
    scale = 20 # 这个值越大，检测到的直线越多

    # 识别竖线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    # 竖直方向上线条获取的步骤同上，唯一的区别在于腐蚀膨胀的区域为一个宽为1，高为缩放后的图片高度的一个竖长形直条
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=2)
    # cv2.imshow("Dilated row", dilatedrow)
    return dilatedrow

def get_transverse_line(binary):
    rows, cols = binary.shape
    scale = 20 # 这个值越大，检测到的直线越多

    # 识别横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    # getStructuringElement： Returns a structuring element of the specified size and shape for morphological operations.
    #  (cols // scale, 1) 为了获取横向的表格线，设置腐蚀和膨胀的操作区域为一个比较大的横向直条
    eroded = cv2.erode(binary, kernel, iterations=1)
    # cv2.imshow("Eroded Image",eroded)
    dilatedcol = cv2.dilate(eroded, kernel, iterations=2)
    # cv2.imshow("Dilated col", dilatedcol)
    return dilatedcol

def bin_img(image):
    """
    对图像进行二值化处理
    :param img: 传入的图像对象（numpy.ndarray类型）
    :return: 二值化后的图像
    """
    # 二值化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255,  # ~取反，很重要，使二值化后的图片是黑底白字
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
    return binary


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
    print("*"*50)
    print("contours: \n")
    for i in range(len(contours) - 1):
        cnt = contours[i]
        print(i,cnt)
        contours_poly[i] = cv2.approxPolyDP(curve=cnt, epsilon=1, closed=True)
        # 以指定的精度近似多边形曲线。
        '''    
        .   @param curve Input vector of a 2D point stored in std::vector or Mat
        .   @param epsilon Parameter specifying the approximation accuracy. This is the maximum distance
        .   between the original curve and its approximation.
        .   @param closed If true, the approximated curve is closed (its first and last vertices are
        .   connected). Otherwise, it is not closed.'''
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        rois.append(np.array(boundRect[i]))
        pt1 = (boundRect[i][0], boundRect[i][1]),
        pt2 = (boundRect[i][2], boundRect[i][3]),
        print(img.shape)
        print("pt1:",pt1)
        print("pt2:",pt2)
        img = cv2.rectangle(img_bak,
                            pt1=(boundRect[i][0], boundRect[i][1]),
                            pt2=(boundRect[i][2], boundRect[i][3]),
                            color=(0, 0, 255),
                            thickness=2,
                            lineType=1,
                            shift=0)
    cv2.imshow("contour",img)
    rois = split_rec(rois)
    return rois

if __name__ == "__main__":
    image  = "./img/table-6.png"
    image1  = "./img/9.jpg"

    img_bak = cv2.imread(image)
    img = bin_img(img_bak)

    # img_transverse = erode_img(img,(1,2),40)
    # img_vertical = erode_img(img, (2,1), 40)
    # # img = img_transverse + img_vertical
    # img_transverse = dilate_img(img_transverse,(2,2),1)
    # img_vertical = dilate_img(img_vertical,(2,2),1)
    #
    # img = get_points(img_transverse,img_vertical)

    dilatedcol, dilatedrow = get_vertical_line(img), get_transverse_line(img)
    img = get_points(dilatedcol, dilatedrow)

    rois = get_rec(img)
    print("*"*50)

    print(rois)
    for i, r in enumerate(rois):
        cv2.imshow(str(i), img_bak[r[3]:r[1], r[2]:r[0]])
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    pass

