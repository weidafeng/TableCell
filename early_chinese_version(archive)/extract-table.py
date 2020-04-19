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
# -------------------------------------------------------------------------------
import cv2
import json
import numpy as np
from pathlib import Path
import progressbar


def iter_all_files(folder_dir):
    '''
    遍历文件夹里所有文件，
    过滤掉其他文字（如俄罗斯文）
    输入示例：
    ROOT_DIR = Path('..')
    IMAGE_DIR = ROOT_DIR / Path('img')
    iter_all_files(IMAGE_DIR)

    :param folder_dir: 输入文件夹路径
    :return: 文件夹内所有文件名的列表（只返回jpg文件）
    '''
    capital = [chr(x) for x in range(65,91)]
    lowercase = [chr(x) for x in range(97, 123)]
    capital.extend(lowercase)
    im_files = [f for f in folder_dir.iterdir() if f.suffix == '.jpg' and f.stem[0] in capital]
    # im_files.sort(key=lambda f: int(f.stem[1:]),reverse=True)  # 排序，防止顺序错乱、数据和标签不对应
    # print("length:",len(im_files),"\n im_files：",im_files)

    # 进度条
    # w = progressbar.widgets
    # widgets = ['Progress: ', w.Percentage(), ' ', w.Bar('#'), ' ', w.Timer(),
    #            ' ', w.ETA(), ' ', w.FileTransferSpeed()]
    # progress = progressbar.ProgressBar(widgets=widgets)
    # for im_file in progress(im_files):
    #
    #     print(im_file)
    return im_files


def get_rec(img):
    """
    获取单元格顶点坐标
    :param img:
    :return:
    """
    # 在mask那张图上通过findContours 找到轮廓，判断轮廓形状和大小是否为表格。
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_poly = [0] * len(contours)
    # print("len contours",len(contours))
    boundRect = [0] * len(contours)
    rois = []
    rois_list = []
    for i in range(len(contours)):
        cnt = contours[i]
        # approxPolyDP 函数用来逼近区域成为一个形状，true值表示产生的区域为闭合区域。
        contours_poly[i] = cv2.approxPolyDP(cnt, 2, True)
        # boundingRect为将这片区域转化为矩形，此矩形包含输入的形状。
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        rois.append(np.array(boundRect[i]))
        rois_list.append(list(boundRect[i]))
        # img = cv2.rectangle(img_bak, (boundRect[i][0], boundRect[i][1]), (boundRect[i][2], boundRect[i][3]),
        #                     (255, 255, 255), 1, 8, 0)
    # rois = split_rec(rois)
    # print("len rois", len(rois_list))
    return rois_list, rois


def get_total_row_cols(x):
    '''
    # # 输入交点列表，计算每行一共有多少个点
    # 输出为点的行偏移、本行点数（字典形式）
    # 格式
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
    # 输入一个列表，指定一个精度，key之间小于该精度的，归为一类
    # err = 2  # 允许的误差
    '''
    # 例如本例，452和451相近，归为一类
    # d = {770: 5, 730: 5, 683: 5, 644: 5, 617: 5, 471: 3, 470: 2, 452: 3, 451: 2, 414: 5, 360: 5, 286: 5, 50: 5}
    '''
    d = row  # 输入的字典（横坐标：该行点数）
    d_keys = list(d.keys())
    for i in range(len(d_keys) - 1):
        #     print(d_keys[i],d_keys[i+1])
        if abs(d_keys[i + 1] - d_keys[i]) < err:  # 两个点在误差允许范围内很接近
            #         print(d[d_keys[i]] + d[d_keys[i+1]])  # 两点总数合并
            d[d_keys[i + 1]] = d[d_keys[i]] + d[d_keys[i + 1]]  # 两点总数合并
            del d[d_keys[i]]  # 删除其中一个
    # print(d)
    return d  # 清洗后的字典{横坐标：该行点数}


def get_dots(x, row):
    # 得到点的坐标
    # 输入：
    #   点列表x，
    #   每行点数
    results = []
    # print("坐标值， 本行点数")
    for key in row:
        #     print(row[key])
        #     print("*"*50)
        #     print(key, row[key])
        for val in range(row[key]):
            #         print(key)
            yy = key
            xx = [val[0] for val in x if val[1] == yy]
            result = [[x, yy] for x in xx]
        # print(result)
        results.append(result)
    return results


def get_bounding_box(results):
    '''
    # 得到bounding box的对角线两点坐标（右下角、左上角）
    # 决定提取单元格效果的关键是设计的人工规则
    :param results: results = get_dots(row)
    :return: 对角两点坐标列表
    '''
    bounding_box = []
    for i in range(len(results) - 1):
        col_down = results[i]
        col_up = results[i + 1]
        #         print(col_down)
        #         print(col_up)
        len_down, len_up = len(col_down), len(col_up)
        #         print(len_down,len_up)

        if len_down == len_up:  # 上下两行点数相同，直接取对角点
            #             print("上下两行点数相同，直接取对角点")
            for j in range(len(col_down) - 1):
                # print(col_down[j], col_up[j + 1])
                bounding_box.append([col_down[j], col_up[j + 1]])
        elif len_down > len_up:  # 下面点数多：
            #             print("下面点数多")
            for j in range(len(col_up) - 1):
                k = j  # k存储多的点
                while k < len_down - 1:  # 遍历下面所有的点（点数多的那条直线）
                    if col_down[k + 1][0] == col_up[j + 1][0] :  # 末尾两点匹配，且开头两点匹配
                        # print(col_down[k], col_up[j + 1])
                        bounding_box.append([col_down[j], col_up[j + 1]])
                    k += 1
        else:  # 上面点数多
            #             print("上面点数多")
            for j in range(len(col_down) - 1):
                k = j  # k存储多的点
                while k < len_up - 1:  # 遍历上面所有的点（点数多的那条直线）
                    if col_up[k + 1][0] == col_down[j + 1][0] and col_down[j][0] in col_up[k+1]:  # 末尾两点匹配，且开头两点匹配
                        # print(col_down[j], col_up[k + 1])
                        bounding_box.append([col_down[j], col_up[k + 1]])
                    k += 1
    return bounding_box


def draw_bbox(img, bboxs, img_name='None', save=False, show=True):
    """
    可视化单元格
    输入：图像、坐标列表
    :param img:
    :param bboxs: 输入的单元格坐标列表，格式：[左下角、右上角]
    :param save:
        True: 保存成图像,图像名为“原图像_box.jpg”
        False: 不保存，只可视化
    :param img_name: 若指定save 为True，则需指定该项
    :return:
    """
    for i in range(len(bboxs)):
        '''
        cv2.rectangle 的两个参数分别代表矩形的左上角和右下角两个点，
        而且 x 坐标轴是水平方向的，y 坐标轴是垂直方向的。

        x1,y1 ------     -> x
        |          |
        |          |
        |          |
        --------x2,y2       
        ∨
        y

        '''
        for i in range(len(bboxs)):
            pt1 = (bboxs[i][1][0], bboxs[i][1][1])  # 左上角
            pt2 = (bboxs[i][0][0], bboxs[i][0][1])  # 右下角

            img = cv2.rectangle(img,
                                pt1=pt1,
                                pt2=pt2,
                                color=(255, 0, 0),
                                thickness=2,
                                lineType=1,
                                shift=0)
    if save:
        # assert img_name != 'None', "如需保存结果，应指定图像名"
        result_name = "./results/" + img_name + ".jpg"
        cv2.imwrite(filename=result_name, img=img)
        output_json = Path('./results_label') / Path(f'{img_name}.json')
        with output_json.open('w', encoding='utf-8') as f:
            json.dump(bboxs, f)

    if show:  # 可视化
        cv2.imshow("contour", img)


def extract_lines(image, scale=20, erode_iters=1, dilate_iters=2, show=True):
    # 输入一张图片，提取横线、竖线
    '''
    :param image:  image = cv2.imread(img_path,1)
    :param scale:     scale = 20 # 这个值越大，检测到的直线越多
    :param erode_iters: 腐蚀的次数
    :param dilate_iters: 膨胀的次数
    :param show 是否可视化
    :return: dilatedcol, dilatedrow : 得到的竖线、横线
    '''

    # 二值化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255,  # ~取反，很重要，使二值化后的图片是黑底白字
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
    # cv2.imshow("binary ", binary)

    rows, cols = binary.shape

    # 识别横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    # getStructuringElement： Returns a structuring element of the specified size and shape for morphological operations.
    #  (cols // scale, 1) 为了获取横向的表格线，设置腐蚀和膨胀的操作区域为一个比较大的横向直条
    eroded = cv2.erode(binary, kernel, iterations=erode_iters)
    # cv2.imshow("Eroded Image",eroded)
    dilatedcol = cv2.dilate(eroded, kernel, iterations=dilate_iters)  # 为了是表格闭合，故意使得到的横向更长（以得到交点——bounding-box）

    # 识别竖线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    # 竖直方向上线条获取的步骤同上，唯一的区别在于腐蚀膨胀的区域为一个宽为1，高为缩放后的图片高度的一个竖长形直条
    eroded = cv2.erode(binary, kernel, iterations=erode_iters)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=dilate_iters)  # 为了是表格闭合，故意使线变长
    if show:
        print("shape:", rows, cols)
        cv2.imshow("Dilated col", dilatedcol)
        cv2.imshow("Dilated row", dilatedrow)
        # 绘制出横线、竖线
        merge = cv2.add(dilatedcol, dilatedrow)
        cv2.imshow("col & row", merge)
    return dilatedcol, dilatedrow


def get_bit_wise(col, row, show=True):
    '''
    输入横线、竖线，得到交点
    :param col: 竖线
    :param row: 横线
    :return:
    '''
    # 标识交点
    bitwiseAnd = cv2.bitwise_and(col, row)
    if show:
        cv2.imshow("bitwiseAnd Image", bitwiseAnd)
    return bitwiseAnd


def process_single_image(img_path, show=True, save=False, scale=20, erode_iters=1, dilate_iters=2):
    '''
    输入单个图像路径
    1. 提取表格线
    2. 得到横线、竖线的交点
    3. 通过交点找到矩形单元格坐标
    4. 计算每一行有多少点
    5. 清洗合并点（有些横不平、竖不直，只差一两个像素）
    6. 把点转化成bounding box格式（左上角、右下角）
    7. 可视化
    :param img_path: 非字符串格式，是pathlib.Path('./img')格式，方便后续提取图像名、保存结果
    :param scale=20, 越大提取的线越多
    :param  erode_iters=1
    :param  dilate_iters=2
    :param
    :return:
    '''
    img_name = img.stem  # 用于保存结果
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


    # 绘制单元格，save=False，可视化
    # save = True，指定img_name,保存图像
    draw_bbox(image, bounding_boxs, img_name=img_name, save=save, show=show)

    if show:
        print("bounding_boxs:",bounding_boxs)
        print("len(bounding_boxs):", len(bounding_boxs))
        cv2.imshow("img", image)
        cv2.waitKey(0)


if __name__ == '__main__':
    ROOT_DIR = Path('./img1')
    ROOT_DIR= Path('E:/dataset/TableBank/Word/train2017')
    imgs_list = iter_all_files(ROOT_DIR)

    w = progressbar.widgets
    widgets = ['Progress: ', w.Percentage(), ' ', w.Bar('#'), ' ', w.Timer(),
               ' ', w.ETA(), ' ', w.FileTransferSpeed()]
    progress = progressbar.ProgressBar(widgets=widgets)
    for img in progress(imgs_list):
        print(img)
        # process_single_image(img,show=True,save=False)
        process_single_image(img, show=False, save=True)

# 参考： https://blog.csdn.net/yomo127/article/details/52045146
