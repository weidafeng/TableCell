# -*- coding: utf-8 -*-
'''
先用cv2.getPerspectiveTransform（target_points, four_points）得到旋转矩阵
然后再用cv2.warpPerspective(img, M, (weight, height))进行透视变换。

关键是找到四个角点，本代码人工指定角点坐标
'''

import cv2
import numpy as np
img= cv2.imread('img/rowRotate.jpg')#读取原始图片
cv2.imshow("raw",img)
target_points =[[278,189],[758,336],[570,1034],[65,900]]
#这里我们通过人工的方式读出四个角点A，B，C，D
height = img.shape[0]
weight = img.shape[1]
four_points= np.array(((0, 0),
                       (weight - 1, 0),
                       (weight - 1, height - 1),
                       (0, height - 1)),
                    np.float32)
target_points = np.array(target_points, np.float32)#统一格式
M = cv2.getPerspectiveTransform(target_points, four_points)
Rotated= cv2.warpPerspective(img, M, (weight, height))
cv2.imshow("Rotated.jpg",Rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

