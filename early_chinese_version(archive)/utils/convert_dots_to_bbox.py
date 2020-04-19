# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         convert_dots_to_bbox.py
# Author:       wdf
# Date:         2019/7/18
# IDE：         PyCharm 
# Parameters:
#     @param:
#     @param:
# Return： 
#       
# Description:  
# Usage：
#-------------------------------------------------------------------------------


def get_total_row_cols(x):
    # # 输入交点列表，计算每行一共有多少个点
    # 输出为点的行偏移、本行点数（字典形式）
    # 格式
    #  [58, 174, 1, 1],
    #  [557, 145, 1, 1],
    #  [513, 145, 1, 1],
    #  [471, 145, 1, 1],
    #  [58, 145, 1, 1]]

    row = {}
    num = 1
    for i in range(len(x)-1):
        if x[i][1] == x[i+1][1]:
            num += 1
            row[x[i][1]] = num
        else:
            num = 1
    return row

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
            xx = [val[0] for val in x if val[1]==yy]
            result = [[x,yy] for x in xx]
        # print(result)
        results.append(result)
    return results


def get_bounding_box(results):
    # 得到bounding box的对角线两点坐标（右下角、左上角）
    # 输入：results = get_dots(row)
    # 输出：

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
                    if col_down[k + 1][0] == col_up[j + 1][0]:
                        # print(col_down[k], col_up[j + 1])
                        bounding_box.append([col_down[k], col_up[j + 1]])
                    k += 1
        else:  # 上面点数多
            #             print("上面点数多")
            for j in range(len(col_down) - 1):
                k = j  # k存储多的点
                while k < len_up - 1:  # 遍历上面所有的点（点数多的那条直线）
                    if col_up[k + 1][0] == col_down[j + 1][0]:
                        # print(col_down[j], col_up[k + 1])
                        bounding_box.append([col_down[j], col_up[k + 1]])
                    k += 1
    return bounding_box


def main():
    x = [[549, 764, 1, 1], [317, 764, 1, 1], [85, 764, 1, 1], [549, 738, 1, 1], [317, 738, 1, 1], [85, 738, 1, 1],
         [549, 712, 1, 1], [317, 712, 1, 1], [85, 712, 1, 1], [549, 687, 1, 1], [317, 687, 1, 1], [85, 687, 1, 1],
         [549, 636, 1, 1], [317, 636, 1, 1], [85, 636, 1, 1], [549, 539, 1, 1], [317, 539, 1, 1], [85, 539, 1, 1],
         [549, 488, 1, 1], [317, 488, 1, 1], [85, 488, 1, 1], [549, 462, 1, 1], [317, 462, 1, 1], [85, 462, 1, 1],
         [549, 343, 1, 1], [85, 343, 1, 1], [549, 317, 1, 1], [317, 317, 1, 1], [85, 317, 1, 1], [549, 279, 1, 1],
         [317, 279, 1, 1], [85, 279, 1, 1], [549, 253, 1, 1], [317, 253, 1, 1], [85, 253, 1, 1], [85, 82, 1, 1],
         [85, 69, 1, 1]]

    row = get_total_row_cols(x)
    results = get_dots(x, row)
    # print(results)

    bounding_boxs = get_bounding_box(results)
    print(bounding_boxs)

if __name__ == '__main__':
    main()