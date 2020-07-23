from mmdet.apis import init_detector, inference_detector,async_inference_detector
import mmcv

from mmdet.core import get_classes
import os
import numpy as np
import cv2
import pycocotools.mask as maskutils
# from osgeo import gdal


# def GeotiffR(filename):
#     dataset = gdal.Open(filename)
#     im_porj = dataset.GetProjection()
#     im_geotrans = dataset.GetGeoTransform()
#     im_data = np.array(dataset.ReadAsArray(), dtype='int8')
#     im_shape = im_data.shape
#     del dataset
#     return im_data, im_porj, im_geotrans, im_shape

# def GeotiffW(filename, im_shape, single, im_geotrans, im_porj):
#     datatype = gdal.GDT_Byte
#     driver = gdal.GetDriverByName("GTiff")
#     dataset = driver.Create(filename, im_shape[1], im_shape[0], 1, datatype)
#     dataset.SetGeoTransform(im_geotrans)
#     dataset.SetProjection(im_porj)
#     dataset.GetRasterBand(1).WriteArray(single)
#     del dataset

def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=2):
    BOX_COLOR = (255, 0, 0)
    TEXT_COLOR = (255, 255, 255)
    if bbox.size:
        for singbbox in bbox:
            score = singbbox[-1]
            rbbox = [int(x) for x in singbbox[:-1]]
            x_min, y_min, x_max, y_max = rbbox
            class_name = class_name+ ":" + str(score)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
            ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
            cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
            cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    else:
        img = img
    return img

def visualize(img, bboxes):
    labeindex = ['ssc','msc','bssc','bsc','rc']
    for label in labeindex:
        i = 0 
        img = visualize_bbox(img, bboxes[i], label)
        i = i+1
    return img

def bbox_transfrom(inix ,iniy, bboxes):
    for i in range(len(bboxes)):
        if bboxes[i].size:
            for j in range(len(bboxes[i])):
                bboxes[i][j][0] = int(bboxes[i][j][0])+ inix
                bboxes[i][j][1] = int(bboxes[i][j][1])+ iniy
                bboxes[i][j][2] = int(bboxes[i][j][2])+ inix
                bboxes[i][j][3] = int(bboxes[i][j][3])+ iniy
    return bboxes

config_file = 'configs/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco.py'
checkpoint_file = '/data/mmdetection/work_dirs/faster_rcnn_x101_32x4d_fpn_1x_coco/latest.pth'
image_size = [448,448]

# build the model from a config file and a checkpoint file
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
model = init_detector(config_file, checkpoint_file)


source = 'UAV2.jpg'
GeoType = False
if source.split('.')[1] == 'tif':
    Geodata = gdal.Open(source)
    GeoType = True
    if not dataset:
        print(source, " :文件无法打开") 
    im_width =dataset.RasterXSize                                                       # 列数
    im_height = dataset.RasterYSize                                                     # 行数
    print(im_width, im_height)
    PredImg = Geodata
else:
    PredImg = cv2.imread(source) # print(type(im))
    im_height = PredImg.shape[0]
    im_width = PredImg.shape[1]
    print(im_width, im_height)

viewImg= np.zeros((image_size[0],image_size[1], 3))

SavePath = 'pred/'
new_name = len(os.listdir(SavePath)) + 1
bboxouts = []
for imgi in range(image_size[0]*8, im_height,image_size[0]):
    for imgj in range(image_size[0]*8, im_width,image_size[1]):
        cropped = PredImg[imgi:imgi+image_size[0], imgj:imgj+image_size[1],:]
        print(cropped.shape)
        if cropped.shape[0]< image_size[0] or cropped.shape[1]< image_size[1]:
            NewI = np.zeros((image_size[0], image_size[0], 3))
            NewI[:cropped.shape[0],:cropped.shape[1], :] = cropped[:,:,:]
            cropped = NewI
            print('goubi')
        resultbbox = inference_detector(model, cropped)
        gouzi1 = visualize(cropped,resultbbox)
        cv2.imwrite(SavePath + "/%d.png"%new_name,gouzi1)
        bboxTrans = bbox_transfrom(imgi,imgj,resultbbox)
        bboxouts.append(bboxTrans)
        new_name=new_name+1
        break
    break

for bboxout in bboxouts:
    resultImg = visualize(PredImg,bboxout)
# model.show_result(cropped, result, out_file=SavePath + "/%d.png"%new_name)
cv2.imwrite(SavePath + "/gouzi.png",resultImg)
#         # break

# result = inference_detector(model, source)
# model.show_result(source, result, out_file='pred/'+'dd'+'.png')
# for i in img:
#     # test a single image and show the results
#     name = i.split('test/')[1]
#     print(name)
#     img = i # or img = mmcv.imread(img), which will only load it once
#     result = inference_detector(model, img)
#     print(result)
#     # visualize the results in a new window
#     # model.show_result(img, result)

#     # or save the visualization results to image files
#     model.show_result(img, result, out_file='pred/'+name+'.png')
#     # answer = show_mask_result(img, result, 'pred/'+i+'.png',
#     #                         score_thr=mask_score_thresh, with_mask=True)
#     break