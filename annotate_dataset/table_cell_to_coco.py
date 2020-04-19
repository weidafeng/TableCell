# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         table-cell-to-coco.py
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
# -------------------------------------------------------------------------------

import datetime
import json
from pathlib import Path
import re
from PIL import Image
import numpy as np
import progressbar
from multiprocessing import pool
from pycococreatortools import pycococreatortools

ROOT_DIR = Path('/media/tristan/Files/dataset/TableBank/table-cell/new_table_cell/new_table_cell_5262/')
IMAGE_DIR = ROOT_DIR / Path('train2017')
ANNOTATION_DIR = ROOT_DIR / Path('labels_erase_col_lines_2100train')
RESULT_JSON_DIR = ROOT_DIR / Path('instances_train2017.json')

INFO = {
    "description": "TABLE-CELL Dataset",
    "url": "https://github.com/weidafeng/tablecell",
    "version": "0.1.0",
    "year": 2020,
    "contributor": "Onlyyou-SY",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 0,
        'name': 'background',
        'supercategory': 'shape',
    },
    {
        'id': 1,
        'name': 'cell',
        'supercategory': 'shape',
    },
]


# bounding-box and segmentation information
def get_info(content):
    left, top = float(content[1][0]), float(content[1][1])
    right, down = float(content[0][0]), float(content[0][1])

    height = down - top
    width = right - left
    segmentation = [left, top, left + width, top, left + width, top + height, left, top + height]
    # coco format:
    # bounding-box,  (x,y,w,h）
    # segmentation: [[1,2,3,4,5,6,7,8]]  # points in order
    return [left, top, width, height], [segmentation]


def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    annotation_id = 1

    im_files = [f for f in IMAGE_DIR.iterdir()]
    im_files.sort(key=lambda f: f.stem, reverse=True)

    an_files = [f for f in ANNOTATION_DIR.iterdir()]
    an_files.sort(key=lambda f: f.stem, reverse=True)

    assert len(an_files) == len(im_files), \
        "#images does not equal to #labels, please run diff_two_folder.py，and delete the mis-match file."

    for im_file, an_file in zip(im_files, an_files):
        image = Image.open(im_file)
        im_info = pycococreatortools.create_image_info(image_id, im_file.name, image.size)
        coco_output['images'].append(im_info)
        myPool = pool.Pool(processes=16)

        annotation_info_list = []

        with open(an_file, 'r') as f:
            datas = json.load(f)
            for i in range(len(datas)):
                data = datas[i]
                # print(data)
                bounding_box = get_info(data)[0]
                segmentation = get_info(data)[1]

                class_id = 1

                print(bounding_box, segmentation)
                area = bounding_box[-1] * bounding_box[-2]
                an_infos = pycococreatortools.mask_create_annotation_info(annotation_id=annotation_id,
                                                                          image_id=image_id, category_id=class_id,
                                                                          area=area, image_size=image.size,
                                                                          bounding_box=bounding_box,
                                                                          segmentation=segmentation)
                annotation_info_list.append(an_infos)
                annotation_id += 1

        myPool.close()
        myPool.join()

        for annotation_info in annotation_info_list:
            if annotation_info is not None:
                coco_output['annotations'].append(annotation_info)
        image_id += 1

    print("[INFO]: Saving annotations")
    output_json = Path(RESULT_JSON_DIR)
    with output_json.open('w', encoding='utf-8') as f:
        json.dump(coco_output, f)
    print("[INFO]: Annotations JSON file saved in：", str(output_json))


if __name__ == "__main__":
    main()
