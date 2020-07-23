import argparse
import glob
import os.path as osp

# import cityscapesscripts.helpers.labels as CSLabels
import mmcv
import numpy as np
import pycocotools.mask as maskUtils

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
# from pycococreatortools import pycococreatortools
import cococreate as pycococreatortools


INFO = {
    "description": "Water Dataset",
    "version": "0.1.0",
    "year": 2020,
    "contributor": "HaoweiGis",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Haowei",
        "url": "http://www.deekong.com/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'water',
        'supercategory': 'shape',
    }]


def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files

def collect_annotations(files, nproc=1):
    print('Loading annotation images')
    if nproc > 1:
        images = mmcv.track_parallel_progress(
            img2coco, files, nproc=nproc)
    else:
        images = mmcv.track_progress(img2coco, files)

    return images


def img2coco(ROOT_DIR):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
 
    image_id = 1
    segmentation_id = 1

    # filter for jpeg images
    IMAGE_DIR = os.path.join(ROOT_DIR, "image")
    ANNOTATION_DIR = os.path.join(ROOT_DIR, "label")
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:
                    
                    if 'gt' in annotation_filename:
                        class_id = 1
                    # elif 'circle' in annotation_filename:
                    #     class_id = 2
                    # else:
                    #     class_id = 3

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                        ).astype(np.uint8)
                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)
                    # print(annotation_info)
                    if annotation_info is not None:
                        for annotation_single in annotation_info:
                            annotation_single["id"] = segmentation_id
                            coco_output["annotations"].append(annotation_single)
                            segmentation_id = segmentation_id + 1
                  
                image_id = image_id + 1
    return coco_output

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert custom data annotations to COCO format')
    parser.add_argument('work_path', help='cityscapes data path')
    # parser.add_argument('--img-dir', default='leftImg8bit', type=str)
    # parser.add_argument('--label-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    work_path = args.work_path
    out_dir = args.out_dir if args.out_dir else work_path
    mmcv.mkdir_or_exist(out_dir)

    set_name = dict(
        train='instancesonly_filtered_train.json',
        val='instancesonly_filtered_val.json',
        # test='instancesonly_filtered_test.json'
        )

    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with mmcv.Timer(
                print_tmpl='It tooks {}s to convert coco annotation'):
            out_json = img2coco(
                osp.join(work_path, split))    
        #     with open('json_name'.format(ROOT_DIR), 'w') as output_json_file:
        # json.dump(coco_output, output_json_file)
            mmcv.dump(out_json, osp.join(out_dir, json_name))
        # break


if __name__ == '__main__':
    main()


# python tools/convert_datasets/img2coco.py ./data/builidingGF2/ --out-dir ./data/builidingGF2/annotations/
# python img2coco.py /data/InstanceSeg/Semantic/ImageDataGF2 --out-dir /data/InstanceSeg/Semantic/ImageDataGF2/annotations/