# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         extract-table.py
# Author:       wdf
# Date:         2019/7/9
# IDE：         PyCharm 
# Parameters:
#     @param:
#     @param:
# Return：
#
# Description: extract lines in table
# Reference：https://answers.opencv.org/question/63847/how-to-extract-tables-from-an-image/
# -------------------------------------------------------------------------------

import cv2
import json
import numpy as np
from pathlib import Path
import progressbar


def iter_all_files(folder_dir):
    '''
    Filter out names that are not standardized.
    :param folder_dir: Path()
    :return: all jpg images
    '''
    capital = [chr(x) for x in range(65, 91)]
    lowercase = [chr(x) for x in range(97, 123)]
    capital.extend(lowercase)
    im_files = [f for f in folder_dir.iterdir() if f.suffix == '.jpg' and f.stem[0] in capital]
    return im_files


def get_rec(img):
    """
    Gets the vertex coordinates of the cells
    :param img:
    :return:
    """
    # Find the contour on the mask map and determine whether the contour is a table by shape and size .
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_poly = [0] * len(contours)
    # print("len contours",len(contours))
    boundRect = [0] * len(contours)
    rois = []
    rois_list = []
    for i in range(len(contours)):
        cnt = contours[i]
        contours_poly[i] = cv2.approxPolyDP(cnt, 2, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        rois.append(np.array(boundRect[i]))
        rois_list.append(list(boundRect[i]))
    return rois_list, rois


def get_total_row_cols(x):
    '''
    input the intersection list and calculate how many points per row
    # format:
    #  [58, 174, 1, 1],
    #  [557, 145, 1, 1],
    #  [513, 145, 1, 1],
    #  [471, 145, 1, 1],
    #  [58, 145, 1, 1]]
    :param x:
    :return:
    '''

    row = {}
    num = 1
    for i in range(len(x) - 1):
        if x[i][1] == x[i + 1][1]:
            num += 1
            row[x[i][1]] = num
        else:
            num = 1
    return row


def clean_dots(row, err=1):
    '''
    # for example，452 is close to451 ，so merge to one class
    # d = {770: 5, 730: 5, 683: 5, 644: 5, 617: 5, 471: 3, 470: 2, 452: 3, 451: 2, 414: 5, 360: 5, 286: 5, 50: 5}

    :param row: a dict
    :param err: control the precision
    :return:
    '''
    d = row
    d_keys = list(d.keys())
    for i in range(len(d_keys) - 1):
        if abs(d_keys[i + 1] - d_keys[i]) < err:
            d[d_keys[i + 1]] = d[d_keys[i]] + d[d_keys[i + 1]]  # 两点总数合并
            del d[d_keys[i]]  # 删除其中一个
    return d


def get_dots(x, row):
    results = []
    for key in row:
        for val in range(row[key]):
            yy = key
            xx = [val[0] for val in x if val[1] == yy]
            result = [[x, yy] for x in xx]
        results.append(result)
    return results


def get_bounding_box(results):
    '''
    Some manual rules to determine whether the bbox belongs to a table.
    There is room for optimization.

    :param results: results = get_dots(row)
    :return:
    '''
    bounding_box = []
    for i in range(len(results) - 1):
        col_down = results[i]
        col_up = results[i + 1]
        len_down, len_up = len(col_down), len(col_up)

        if len_down == len_up:
            for j in range(len(col_down) - 1):
                bounding_box.append([col_down[j], col_up[j + 1]])
        elif len_down > len_up:
            for j in range(len(col_up) - 1):
                k = j
                while k < len_down - 1:
                    if col_down[k + 1][0] == col_up[j + 1][0]:
                        bounding_box.append([col_down[j], col_up[j + 1]])
                    k += 1
        else:
            for j in range(len(col_down) - 1):
                k = j
                while k < len_up - 1:
                    if col_up[k + 1][0] == col_down[j + 1][0] and col_down[j][0] in col_up[k + 1]:
                        bounding_box.append([col_down[j], col_up[k + 1]])
                    k += 1
    return bounding_box


def draw_bbox(img, bboxs, img_name='None', save=False, show=True):
    """
    visualization
    :param img:
    :param bboxs: 输入的单元格坐标列表，格式：[左下角、右上角]
    :param save: whether to save the result image
    :param img_name: should be specified if save is set True.
    :return:
    """
    for i in range(len(bboxs)):
        '''
        cv2.rectangle 

        x1,y1 -----------  -> x
        |          |
        |          |
        |          |
        | --------x2,y2       
        |
        ∨
        y

        '''
        for i in range(len(bboxs)):
            pt1 = (bboxs[i][1][0], bboxs[i][1][1])  # left top
            pt2 = (bboxs[i][0][0], bboxs[i][0][1])  # right bottom

            img = cv2.rectangle(img,
                                pt1=pt1,
                                pt2=pt2,
                                color=(255, 0, 0),
                                thickness=2,
                                lineType=1,
                                shift=0)
    if save:
        result_name = "./results/" + img_name + ".png"
        cv2.imwrite(filename=result_name, img=img)
        output_json = Path('./results_label') / Path(f'{img_name}.json')
        with output_json.open('w', encoding='utf-8') as f:
            json.dump(bboxs, f)

    if show:
        cv2.imshow("contour", img)


def extract_lines(image, scale=20, erode_iters=1, dilate_iters=2, show=True):
    '''
    extract col and row lines
    :param image:
    :param scale:
    :param erode_iters:
    :param dilate_iters:
    :param show:
    :return: dilatedcol, dilatedrow
    '''
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

    if show:
        print("shape:", rows, cols)
        cv2.imshow("Dilated col", dilatedcol)
        cv2.imshow("Dilated row", dilatedrow)
        merge = cv2.add(dilatedcol, dilatedrow)
        cv2.imshow("col & row", merge)
    return dilatedcol, dilatedrow


def get_bit_wise(col, row, show=True):
    bitwiseAnd = cv2.bitwise_and(col, row)
    if show:
        cv2.imshow("bitwiseAnd Image", bitwiseAnd)
        # cv2.imwrite('bitwise_add.png', bitwiseAnd)
    return bitwiseAnd


def process_single_image(img_path, show=True, save=False, scale=20, erode_iters=1, dilate_iters=2):
    '''
    Input a single image path
    1. Extract table lines
    2. Get the intersection of horizontal and vertical lines
    3. Find the coordinates of the rectangular cells by the intersection point
    4. Calculate how many points are in each row
    5. Clean the merged points (manaual rules)
    6. Convert points to bounding box format (upper left, lower right)
    7. Visualize
    '''
    img_name = img.stem
    img_path = str(img_path)
    image = cv2.imread(img_path, 1)

    dilatedcol, dilatedrow = extract_lines(image, scale=scale, erode_iters=erode_iters, dilate_iters=dilate_iters,
                                           show=show)

    bitwiseAnd = get_bit_wise(col=dilatedcol, row=dilatedrow, show=show)
    rois_list, rois = get_rec(bitwiseAnd)
    # print(rois_list)
    # print(len(rois_list))
    row = get_total_row_cols(x=rois_list)
    row = clean_dots(row)
    results = get_dots(x=rois_list, row=row)

    bounding_boxs = get_bounding_box(results)

    draw_bbox(image, bounding_boxs, img_name=img_name, save=save, show=show)

    if show:
        print("bounding_boxs:", bounding_boxs)
        print("len(bounding_boxs):", len(bounding_boxs))
        cv2.imshow("img", image)
        cv2.waitKey(0)


if __name__ == '__main__':
    ROOT_DIR = Path('./img2')
    # ROOT_DIR= Path('E:/dataset/TableBank/Word/train2017')
    imgs_list = iter_all_files(ROOT_DIR)

    w = progressbar.widgets
    widgets = ['Progress: ', w.Percentage(), ' ', w.Bar('#'), ' ', w.Timer(), ' ', w.ETA(), ' ', w.FileTransferSpeed()]
    progress = progressbar.ProgressBar(widgets=widgets)
    for img in progress(imgs_list):
        print(img)
        # process_single_image(img,show=True,save=False)
        process_single_image(img, show=True, save=True)

