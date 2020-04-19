# test and save results
# Usage: python test_and_save_results.py -c config/xxx.py -m work_dirs/xxx.pth -i /path/to/test_images -o /path/to/save_results
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configs/xxx.py")
parser.add_argument("-m", "--checkpoint", help="work_dirs/xxx.pth")
parser.add_argument("-i", "--image", help="/path/to/your_test_images")
parser.add_argument("-o", "--output", help="/path/to/save_results")

args = parser.parse_args()

if not os.path.exists(args.output):
    os.mkdir(args.output)

# build the model from a config file and a checkpoint file
model = init_detector(args.config, args.checkpoint, device='cuda:0')

for img in os.listdir(args.image):
    result_path = args.output + '/' + str(args.config.split('/')[-1].split('.')[0]) + '_' + str(img)  # result img path
    img = os.path.join(args.image, img)  # abs path of the input image
    print("[INFO]: Working on ", img)
    result = inference_detector(model, img)
    show_result(img, result, ['cell', 'background'], out_file=result_path, show=False, score_thr=0.3)
    # show_result(img, result, model.CLASSES, out_file=result_path, show=False, score_thr=0.1)
print("[INFO]: Done!")
