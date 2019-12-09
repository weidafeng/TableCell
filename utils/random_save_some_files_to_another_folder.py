# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         random_save_some_files_to_another_folder.py
# Author:       wdf
# Date:         2019/7/20
# IDE：         PyCharm 
# Parameters:
#     @param:
#     @param:
# Return： 
#       
# Description:  
# Usage：
#-------------------------------------------------------------------------------

import random
import os
import shutil

def random_copyfile(srcPath, dstPath, numfiles, move_or_cpoy='move'):
    name_list=list(os.path.join(srcPath,name) for name in os.listdir(srcPath))
    random_name_list=list(random.sample(name_list,numfiles))
    # if not os.path.exists(dstPath):
    #     os.mkdir(dstPath)

    if move_or_cpoy=='move':
        for oldname in random_name_list:
            shutil.move(oldname,oldname.replace(srcPath, dstPath))
    elif move_or_cpoy=='copy':
        for oldname in random_name_list:
            shutil.copyfile(oldname,oldname.replace(srcPath, dstPath))
    else:
        return -1


def main():
    srcPath = 'E:/dataset/TableBank/table-cell/images'
    dstPath = 'E:/dataset/TableBank/table-cell/test_images'
    random_copyfile(srcPath, dstPath, 1000)


if __name__ == '__main__':
    main()