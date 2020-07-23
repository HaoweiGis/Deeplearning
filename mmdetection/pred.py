from mmdet.apis import init_detector, inference_detector,async_inference_detector
import mmcv

from mmdet.core import get_classes
import os
import numpy as np
import pycocotools.mask as maskutils

# def show_mask_result(img, result, save_img, dataset='coco', score_thr=0.7, with_mask=True):
#     segm_result = None
#     if with_mask:
#         bbox_result, segm_result = result
#     else:
#         bbox_result = result
#     if isinstance(dataset, str):  # add own data label to mmdet.core.class_name.py
#         class_names = get_classes(dataset)
#         # print(class_names)
#     elif isinstance(dataset, list):
#         class_names = dataset
#     else:
#         raise TypeError('dataset must be a valid dataset name or a list'
#                         ' of class names, not {}'.format(type(dataset)))
#     h, w, _ = img.shape
#     img_show = img[:h, :w, :]
#     labels = [
#         np.full(bbox.shape[0], i, dtype=np.int32)
#         for i, bbox in enumerate(bbox_result)
#     ]
#     labels = np.concatenate(labels)
#     bboxes = np.vstack(bbox_result)
#     if with_mask:
#         segms = mmcv.concat_list(segm_result)
#         inds = np.where(bboxes[:, -1] > score_thr)[0]
#         for i in inds:
#             color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
#             mask = maskutils.decode(segms[i]).astype(np.bool)
#             img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
#     result_img = mmcv.imshow_det_bboxes(img_show, bboxes, labels, class_names=class_names,
#                                         score_thr=score_thr, show=False, out_file=save_img)
#     return result_img
# mask_score_thresh = 0.6

# config_file = 'configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py'
# checkpoint_file = '/data/mmdetection/work_dirs/faster_rcnn_r101_fpn_1x_coco/latest.pth'

config_file = '/deepai/mmdetection/work_dirs/DaPeng/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco.py'
checkpoint_file = '/deepai/mmdetection/work_dirs/DaPeng/mask_rcnn_r101_fpn_1x_coco/latest.pth'

# build the model from a config file and a checkpoint file
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
model = init_detector(config_file, checkpoint_file)

img = []
img_folder = '/deepai/mmdetection/data/DaPeng/val'
for single_img in os.listdir(img_folder):
    img.append(os.path.join(img_folder, single_img))

for i in img:
    # test a single image and show the results
    name = i.split('val/')[1]
    print(name)
    img = i # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, img)
    # print(result)
    # print(len(result))
    # visualize the results in a new window
    # model.show_result(img, result)

    # or save the visualization results to image files
    model.show_result(img, result, out_file='pred/'+name+'.png')
    # answer = show_mask_result(img, result, 'pred/'+i+'.png',
    #                         score_thr=mask_score_thresh, with_mask=True)
#     break


# [array([[136.34709   ,   4.4106197 , 172.14424   ,  28.720829  ,
#           0.99664956],
#        [213.58635   ,   5.4519153 , 250.52496   ,  30.110144  ,
#           0.9955942 ]]
#         , dtype=float32), 
# array([], shape=(0, 5), dtype=float32), 
# array([], shape=(0, 5), dtype=float32), 
# array([], shape=(0, 5), dtype=float32), 
# array([[ 0.9853128, 41.836945 , 29.625513 , 83.92183  ,  0.9984824]],
#       dtype=float32)]