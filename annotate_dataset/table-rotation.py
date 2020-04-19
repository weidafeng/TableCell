# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         table-rotation.py
# Author:       wdf
# Date:         2019/7/7
# IDE：         PyCharm 
# Description:
#     1. HoughLines ——> get the rotation angle
#     2. warpAffine ——> affine(rotation)
# -------------------------------------------------------------------------------


import math
import cv2
import numpy as np


def get_rotation_angle(image, show_longest_line=True, show_all_lines=False):
    image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=60)
    lengths = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        lengths.append([x1, y1, x2, y2, length])
        if show_all_lines:
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 2)

    lengths.sort(key=lambda x: x[-1])
    longest_line = lengths[-1]
    print("longest_line: ", longest_line)
    x1, y1, x2, y2, length = longest_line
    if show_longest_line:
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("longest", image)

    angle = math.atan((y2 - y1) / (x2 - x1))
    print("angle-radin:", angle)
    angle = angle * (180 / math.pi)
    print("angle-degree:", angle)
    return angle


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)  # rotation matrix
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Calculate the size of the rotated image (avoid image clipping)
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust rotation matrix (avoid image clipping)
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    print("RotationMatrix2D：\n", M)

    # Affine transformation is performed to obtain the image
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_REPLICATE)


def main():
    img_path = "./img1/IMG_20190723_162452.jpg"
    # img_path = "./img/table-1.png"

    img = cv2.imread(img_path)
    angle = get_rotation_angle(img, show_longest_line=False, show_all_lines=False)

    imag = rotate_bound(img, angle)  # 关键
    # cv2.imshow("raw",img)
    # cv2.imshow('rotated', imag)
    cv2.imwrite('rotated.png', imag)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
