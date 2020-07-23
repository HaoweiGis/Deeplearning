# encoding: utf-8
"""
@version: 3.6
@author: mas
@file: mask_rcnn_eval.py
@time: 2020/6/2 15:03
"""
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from demo.predictor import VisualizationDemo

# constants
WINDOW_NAME = "detections"

# inference
INPUT_IMG_PATH = '/data/detectron2/datasets/penghuquGF/val'
OUTPUT_IMG_PATH = '/data/detectron2/cusrun/penghuquGF/outImg'

# 数据集路径
DATASET_ROOT = '/data/detectron2/datasets/penghuquGF'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'train')
VAL_PATH = os.path.join(DATASET_ROOT, 'val')
TRAIN_JSON = os.path.join(ANN_ROOT, 'instancesonly_filtered_train.json')
VAL_JSON = os.path.join(ANN_ROOT, 'instancesonly_filtered_val.json')

# 数据集类别元数据
DATASET_CATEGORIES = [
    {"name": "building", "id": 1, "isthing": 1, "color": [220, 20, 60]},
    #{"name": "neg", "id": 2, "isthing": 1, "color": [219, 142, 185]},
]

# 数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "train_2019": (TRAIN_PATH, TRAIN_JSON),
    "val_2019": (VAL_PATH, VAL_JSON),
}


def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   metadate=get_dataset_instances_meta(),
                                   json_file=json_file,
                                   image_root=image_root)


def get_dataset_instances_meta():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_dataset_instances(name, metadate, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco",
                                  **metadate)


# 注册数据集和元数据
def plain_register_dataset():
    DatasetCatalog.register("train_2019", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH, "train_2019"))
    MetadataCatalog.get("train_2019").set(thing_classes=["pos", "neg"],
                                          json_file=TRAIN_JSON,
                                          image_root=TRAIN_PATH)
    DatasetCatalog.register("val_2019", lambda: load_coco_json(VAL_JSON, VAL_PATH, "val_2019"))
    MetadataCatalog.get("val_2019").set(thing_classes=["pos", "neg"],
                                        json_file=VAL_JSON,
                                        image_root=VAL_PATH)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()  # 拷贝default config副本
    args.config_file = "/data/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    cfg.merge_from_file(args.config_file)  # 从config file 覆盖配置
    cfg.merge_from_list(args.opts)  # 从CLI参数 覆盖配置

    # 更改配置参数
    cfg.DATASETS.TRAIN = ("train_2019",)
    cfg.DATASETS.TEST = ("val_2019",)
    cfg.DATALOADER.NUM_WORKERS = 1  # 单线程
    cfg.INPUT.MAX_SIZE_TRAIN = 512
    cfg.INPUT.MAX_SIZE_TEST = 512
    cfg.INPUT.MIN_SIZE_TRAIN = (512,)
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 类别数
    cfg.MODEL.WEIGHTS = "/data/detectron2/cusrun/penghuquGF/output/model_final.pth"  # 预训练模型权重
    cfg.SOLVER.IMS_PER_BATCH = 8  # batch_size=2; iters_in_one_epoch = dataset_imgs/batch_size
    ITERS_IN_ONE_EPOCH = int(693 / cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 50) - 1  # 12 epochs
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (30000,)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = 5 * ITERS_IN_ONE_EPOCH - 1

    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",default = 'penghuquGF/outImg/',
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    # 注册数据集
    register_dataset()

    demo = VisualizationDemo(cfg, instance_mode=ColorMode.SEGMENTATION)

    # for path in tqdm.tqdm(args.input, disable=not args.output):
    for imgfile in os.listdir(INPUT_IMG_PATH):

        # use PIL, to be consistent with evaluation
        img_fullName = os.path.join(INPUT_IMG_PATH, imgfile)
        img = read_image(img_fullName, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                imgfile, len(predictions["instances"]), time.time() - start_time
            )
        )

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(imgfile))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)
        else:
            # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            # cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            # if cv2.waitKey(0) == 27:
            #     break  # esc to quit
            print("No window")