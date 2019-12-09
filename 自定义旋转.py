# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         自定义旋转.py
# Author:       wdf
# Date:         2019/7/7
# IDE：         PyCharm 
# Description:
# 输入图像、顺时针旋转的角度，旋转整个图像（解决了旋转后图像缺失的问题）
#     原文：https: // blog.csdn.net / qq_37674858 / article / details / 80708393
# Usage：
#-------------------------------------------------------------------------------
import cv2
import numpy as np

# 输入图像、顺时针旋转的角度，旋转整个图像（解决了旋转后图像缺失的问题）
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the center
    # 旋转中心点，默认为图像中心点
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)  # 得到旋转矩阵
    print(M)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    print(cos,sin)

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH),borderMode=cv2.BORDER_REPLICATE)
    # borderMode=cv2.BORDER_REPLICATE 使用边缘值填充
    #borderValue=(255,255,255)) # 使用常数填充边界（0,0,0）表示黑色

def main():
    image = cv2.imread('./img/table-1.png')
    cv2.imshow("raw", image)
    angle = -13
    imag = rotate_bound(image, angle)
    cv2.imshow('rotated', imag)
    cv2.waitKey()


if __name__ == '__main__':
    main()