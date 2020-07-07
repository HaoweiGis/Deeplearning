#训练Mask-RCNN部分
修改changemaskrcnn.py num_class=类别,运行
修改mask_rcnn_r50_fpn.py  num_classes=类别 2个地方
python tools/train.py configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py
python demo/image_demo.py demo/00000.bmp configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py work_dirs/mask_rcnn_r50_fpn_1x_coco/epoch_10.pth


总体流程：
先确定想要训练的网络， 根据base追溯并修改参数
1. cascade_mask_rcnn_hrnetv2p_w32_20e_coco.py
2. ../cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py 修改其中的datasets
3. ../_base_/models/cascade_mask_rcnn_r50_fpn.py 修改num_classes

python tools/train.py configs/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_coco.py