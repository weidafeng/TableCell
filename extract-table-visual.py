# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         extract-table.py
# Author:       wdf
# Date:         2019/7/9
# IDE：         PyCharm 
# Parameters:
#     @param:
#     @param:
# Return： 
#       
# Description:
# 参考：
# https://answers.opencv.org/question/63847/how-to-extract-tables-from-an-image/  （opencv官方示例）
# https://blog.csdn.net/yomo127/article/details/52045146  （c++版代码）
# https://blog.csdn.net/weixin_34059951/article/details/88151801 （python）
#  输入一张平整的图片，提取横线、竖线、交叉点、绘制表格
#  只对平整的图片有效，旋转表格效果不行

#### 分割单元格步骤
# 1. 读取图像；
# 2. 二值化处理；
# 3. 横向、纵向的膨胀、腐蚀操作，得到横线图img_row和竖线图img_col；
# 4. 得到点图，img_row + img_col=img_dot；
# 5. 得到线图，img_row × img_col=img_line（线图只是拿来看看的，后续没有用到）；
# 6. 浓缩点团到单个像素；
# 7. 开始遍历各行的点，将各个单元格从二值图像上裁剪出来，保存到temp文件夹。
# ---------------------
# 原文：https://blog.csdn.net/muxiong0308/article/details/80969355 （python实现）
# 注释+C++实现：https://blog.csdn.net/yomo127/article/details/52045146 （逐行注释）
# Usage： 若表格内容仍被处理为边框，可以调整腐蚀、膨胀函数的参数，比如调大处理次数（iteration）
#-------------------------------------------------------------------------------
import cv2
import numpy as np

def get_rec(img):
    """
    获取单元格顶点坐标
    :param img:
    :return:
    """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [0] * len(contours)
    boundRect = [0] * len(contours)
    rois = []
    print("contours",contours)
    print("len(contours)",len(contours))
    for i in range(len(contours)):
        cnt = contours[i]
        contours_poly[i] = cv2.approxPolyDP(cnt, 1, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        rois.append(np.array(boundRect[i]))
        # img = cv2.rectangle(img_bak, (boundRect[i][0], boundRect[i][1]), (boundRect[i][2], boundRect[i][3]),
        #                     (255, 255, 255), 1, 8, 0)

    return rois

def main(img_path):
    image = cv2.imread(img_path, 1)
    # 二值化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255,  # ~取反，很重要，使二值化后的图片是黑底白字
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
    cv2.imshow("binary ", binary)

    rows, cols = binary.shape
    scale = 20 # 这个值越大，检测到的直线越多

    # 识别横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    # getStructuringElement： Returns a structuring element of the specified size and shape for morphological operations.
    #  (cols // scale, 1) 为了获取横向的表格线，设置腐蚀和膨胀的操作区域为一个比较大的横向直条
    eroded = cv2.erode(binary, kernel, iterations=1)
    # cv2.imshow("Eroded Image",eroded)
    dilatedcol = cv2.dilate(eroded, kernel, iterations=1)
    # cv2.imshow("Dilated col", dilatedcol)


    # 识别竖线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    # 竖直方向上线条获取的步骤同上，唯一的区别在于腐蚀膨胀的区域为一个宽为1，高为缩放后的图片高度的一个竖长形直条
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=1)
    # cv2.imshow("Dilated row", dilatedrow)


    # 标识交点
    bitwiseAnd = cv2.bitwise_and(dilatedcol, dilatedrow)
    cv2.imshow("bitwiseAnd Image", bitwiseAnd)
    rois = get_rec(bitwiseAnd)
    # print(rois)

    lst = []
    for i, r in enumerate(rois):
        print(i,r)
        # cv2.imshow("src" + str(i), image[r[3]:r[1], r[2]:r[0]])
        lst.append(list(r))

    print(lst)
    # 标识表格
    merge = cv2.add(dilatedcol, dilatedrow)
    cv2.imshow("add Image", merge)
    # cv2.imwrite("./img/mask.jpg", merge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    img_path = './img/4.jpg'
    img_path2 = './img/table-6.png'
    main(img_path)
# 参考： https://blog.csdn.net/yomo127/article/details/52045146

