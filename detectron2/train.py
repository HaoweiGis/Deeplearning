import os
import logging
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA

# haowei
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

import random
import cv2
from detectron2.utils.visualizer import Visualizer

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    args.config_file = configfile
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.OUTPUT_DIR = 'work_dir/'+args.config_file.split('/')[-1].split('.')[0]
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 4  # 单线程

    cfg.SOLVER.IMS_PER_BATCH = 4
    # ITERS_IN_ONE_EPOCH = int(3614 / cfg.SOLVER.IMS_PER_BATCH)
    # cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 80) - 1
    # cfg.SOLVER.BASE_LR = 0.002
    # cfg.SOLVER.MOMENTUM = 0.9
    # cfg.SOLVER.WEIGHT_DECAY = 0.0001
    # cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    # cfg.SOLVER.GAMMA = 0.1
    # cfg.SOLVER.STEPS = (15000,)
    # cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    # cfg.SOLVER.WARMUP_METHOD = "linear"
    # cfg.SOLVER.CHECKPOINT_PERIOD = 5*ITERS_IN_ONE_EPOCH - 1
    # cfg.SOLVER.WARMUP_ITERS = 1000
    # cfg.SOLVER.MAX_ITER = 1500 

    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    # cfg.TEST.EVAL_PERIOD = 500

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":

    # register_coco_instances("my_dataset_train", {}, "datasets/waterbody/annotations/instancesonly_filtered_train.json", "datasets/waterbody/train/image")
    # register_coco_instances("my_dataset_val", {}, "datasets/waterbody/annotations/instancesonly_filtered_val.json", "datasets/waterbody/val/image")
    # # configfile = "/deekongai/data/backupfiles/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    # configfile = "/deekongai/configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"

    # register_coco_instances("my_dataset_train", {}, "datasets/bridge/annotations/instances_train.json", "datasets/bridge/data")
    # register_coco_instances("my_dataset_val", {}, "datasets/bridge/annotations/instances_val.json", "datasets/bridge/data")
    # # configfile = "/deekongai/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    # configfile = "/deekongai/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"

    register_coco_instances("my_dataset_train", {}, "/deekongai/data/0geofen/shipdetection/data/annotations/instances_train.json", "/deekongai/data/0geofen/shipdetection/data/trainimage")
    register_coco_instances("my_dataset_val", {}, "/deekongai/data/0geofen/shipdetection/data/annotations/instances_val.json", "/deekongai/data/0geofen/shipdetection/data/trainimage")
    # configfile = "/deekongai/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    configfile = "/deekongai/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"


    #visualize training data
    my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
    dataset_dicts = DatasetCatalog.get("my_dataset_train")
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite(d["file_name"].split('/')[-1].replace('tif','jpg'),vis.get_image()[:, :, ::-1])

    # # model train
    # args = default_argument_parser().parse_args()
    # print("Command Line Args:", args)
    # launch(
    #     main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args,),
    # )
