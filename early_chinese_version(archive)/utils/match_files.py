# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         match_files.py
# Author:       wdf
# Date:         2019/7/20
# IDE：         PyCharm 
# Parameters:
#     @param:
#     @param:
# Return： 
#       
# Description:  根据图片文件夹下的图片名，找到标签文件夹下同名的标签，移动匹配到的标签到指定文件夹。
# Usage：
#-------------------------------------------------------------------------------

import os
import shutil

# 定义获取文件名的方法
def getFileNames(rootDir, suffix=".jpg"):
    fileNames = []
    # 利用os.walk()函数获取根目录下文件夹名称，子文件夹名称及文件名称
    for dirName, subDirList, fileList in os.walk(rootDir):
        for fname in fileList:
            # 用os.path.split()函数来判断并获取文件的后缀名
            if os.path.splitext(fname)[1] == suffix:
                fileNames.append(fname.split(suffix)[0])
                # fileNames.append(dirName + '/' + fname)
    return fileNames


def match_labels(images,labels,src_path='E:/dataset/TableBank/table-cell/labels',dst_path='E:/dataset/TableBank/table-cell/labels'):
    '''
    根据输入的图片名，找到对应的标签，把标签复制到指定文件夹
    :param images: 图片名列表
    :param labels:  标签名列表
    :param src_path: 标签源文件的路径（用于获取源文件绝对路径）
    :param dst_path: 标签保存的目标路径
    :return:
    '''
    for image in images: # 遍历每个图片名
        index = labels.index(image)  # 获取标签索引
        label_name = str(src_path) + "/" + str(labels[index]) + ".json"  # 找到对应标签文件路径
        # shutil.copy(label_name, dst_path) # 把标签复制到对应文件夹
        shutil.move(label_name, dst_path) # 把标签移动到对应文件夹



def main():
    image_root = 'E:/dataset/TableBank/table-cell/test_images'
    label_root = 'E:/dataset/TableBank/table-cell/labels'
    images = getFileNames(image_root,suffix='.jpg')
    labels = getFileNames(label_root,suffix='.json')
    print(images)
    print(labels)
    match_labels(images, labels,src_path=label_root, dst_path='E:/dataset/TableBank/table-cell/test_labels')



if __name__ == '__main__':
    main()