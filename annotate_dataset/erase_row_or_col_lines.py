#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:           wdf
# datetime:         2/28/2020 10:05 PM
# software:         PyCharm
# project name:     TableBank 
# file name:        erase row or column lines
# description:      
# usage:            

import cv2
import os
from multiprocessing import pool


def erase_all_lines(image_path, root, save_folder=None, erase_row_lines=False, erase_col_lines=True, show=False):
    '''
    :param image_path: single image path
    :param root: root path of input image
    :param save_folder: path to save the result images
    :param erase_row_lines: whether erase row lines
    :param erase_col_lines: whether erase col lines
    :param show:  show or not
    :return:
    '''
    image = cv2.imread(os.path.join(root, image_path), 1)

    # binary
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)

    rows, cols = binary.shape
    scale = 20  # The larger the value, the more lines detected.

    # detect row lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    #  In order to obtain transverse table lines, set the operating area for corrosion and expansion to a large
    #  transverse straight bar, i.e. (cols // scale, 1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedcol = cv2.dilate(eroded, kernel, iterations=1)

    # detect col lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=1)

    # intersection points
    bitwiseAnd = cv2.bitwise_and(dilatedcol, dilatedrow)

    if erase_col_lines and erase_row_lines:
        mask = cv2.add(dilatedcol, dilatedrow)
    elif erase_col_lines:
        mask = dilatedrow
    elif erase_row_lines:
        mask = dilatedcol

    # erase lines
    image_copy = image.copy()
    image_copy[mask > 10] = 255  # lines to white
    image_copy[bitwiseAnd > 10] = 0  # intersection points to black

    if show:
        cv2.imshow("raw image ", image)
        cv2.imshow("extract all lines", mask)
        cv2.imshow("result image without line", image_copy)
        # cv2.imwrite('all-lines.png', mask)
        # cv2.imwrite('erase-row-lines.png', image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(save_folder, image_path), image_copy)
    print('done')


def multiprocess_main(folder):
    '''
    multi processing
    :param folder: path to images
    :return:
    '''
    myPool = pool.Pool(processes=4)  # 并行化处理
    for img in os.listdir(folder):
        print(img)
        myPool.apply_async(func=erase_all_lines, args=(img, "E:/dataset/TableBank/table-cell/New folder/images_after",
                                                       "E:/dataset/TableBank/table-cell/New folder/images_erase_col_lines"))
    myPool.close()
    myPool.join()


def main():
    # test a single image
    erase_all_lines('4.jpg', './img', './results', show=True, erase_col_lines=True, erase_row_lines=False)

    # multiprocessing
    # multiprocess_main("E:/dataset/TableBank/table-cell/New folder/images_after")


if __name__ == '__main__':
    main()
