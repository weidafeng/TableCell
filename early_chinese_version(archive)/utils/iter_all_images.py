# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         iter_all_images.py
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
from pathlib import Path
import progressbar

def iter_all_files(folder_dir):
    im_files = [f for f in folder_dir.iterdir()]
    # im_files.sort(key=lambda f: int(f.stem[1:]),reverse=True)  # 排序，防止顺序错乱、数据和标签不对应
    # print("length:",len(im_files),"\n im_files：",im_files)

    # 进度条
    w = progressbar.widgets
    widgets = ['Progress: ', w.Percentage(), ' ', w.Bar('#'), ' ', w.Timer(),
               ' ', w.ETA(), ' ', w.FileTransferSpeed()]
    progress = progressbar.ProgressBar(widgets=widgets)
    for im_file in progress(im_files):
        print(im_file)

def main():
    ROOT_DIR = Path('..')
    IMAGE_DIR = ROOT_DIR / Path('img')
    iter_all_files(IMAGE_DIR)

if __name__ == '__main__':
    main()