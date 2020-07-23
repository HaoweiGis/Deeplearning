from mmdet.apis import init_detector, inference_detector,async_inference_detector
import mmcv

from mmdet.core import get_classes
import os
import numpy as np
import cv2
from tqdm import tqdm
# from osgeo import gdal

def visualize_bbox(img, bbox, class_name, thickness=2):
    BOX_COLOR = (255, 0, 0)
    TEXT_COLOR = (255, 255, 255)
    if bbox.size:
        for singbbox in bbox:
            score = int(singbbox[-1]*100)
            if score >90:
                rbbox = [int(x) for x in singbbox[:-1]]
                x_min, y_min, x_max, y_max = rbbox
                class_name = class_name+ ":" + str(score)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), BOX_COLOR,thickness=thickness)
                ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
                cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
                cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    else:
        img = img
    return img

def visualizesingle(img, bboxes):
    labeindex = ['ssc','msc','bssc','bsc','rc']
    i = 0 
    for label in labeindex:
        img = visualize_bbox(img, bboxes[i], label)
        i = i+1
    return img

def Categorymerge(bboxes):
    correctbbox = None
    for singlebbox in bboxes:
        if correctbbox is None:
            correctbbox = singlebbox
        else:
            correctbbox[0] = np.vstack((correctbbox[0],singlebbox[0]))
            correctbbox[1] = np.vstack((correctbbox[1],singlebbox[1]))
            correctbbox[2] = np.vstack((correctbbox[2],singlebbox[2]))
            correctbbox[3] = np.vstack((correctbbox[3],singlebbox[3]))
            correctbbox[4] = np.vstack((correctbbox[4],singlebbox[4]))
    return correctbbox

def visualize(img, bboxes):

    labeindex = ['ssc','msc','bssc','bsc','rc']
    for i in range(5):
        print(bboxes[i])
        img = visualize_bbox(img, bboxes[i], labeindex[i])
    return correctbbox

def bbox_transfrom(inix ,iniy, bboxes):
    # print('pianyiqian:',inix,iniy)
    for i in range(len(bboxes)):
        if bboxes[i].size:
            for j in range(len(bboxes[i])):
                bboxes[i][j][0] = int(bboxes[i][j][0])+ iniy
                bboxes[i][j][1] = int(bboxes[i][j][1])+ inix
                bboxes[i][j][2] = int(bboxes[i][j][2])+ iniy
                bboxes[i][j][3] = int(bboxes[i][j][3])+ inix
    return bboxes


def SlideIndex(PredImg,im_height,im_width,model):
    new_name = len(os.listdir(SavePath)) + 1
    bboxouts = []
    for imgi in tqdm(range(0, im_height,image_size[0])):
        for imgj in range(0, im_width,image_size[1]):
            cropped = PredImg[imgi:imgi+image_size[0], imgj:imgj+image_size[1],:]
            if cropped.shape[0]< image_size[0] or cropped.shape[1]< image_size[1]:
                NewI = np.zeros((image_size[0], image_size[0], 3))
                NewI[:cropped.shape[0],:cropped.shape[1], :] = cropped[:,:,:]
                cropped = NewI
            resultbbox = inference_detector(model, cropped)
            
            # # 切片文件保存
            # singleImg = visualize(cropped,resultbbox)         
            # cv2.imwrite(SavePath + "/%d.png"%new_name,singleImg)
            # cv2.destroyAllWindows()

            # bbox偏移
            bboxTrans = bbox_transfrom(imgi,imgj,resultbbox)
            bboxouts.append(resultbbox)

            new_name=new_name+1
    return bboxouts

def ImgRead(imgfile):
    if imgfile.split('.')[1] == 'tif':
        # PredImg = np.transpose(gdal.Open(imgfile).ReadAsArray(),axes=(1, 2, 0))
        Geodata = gdal.Open(imgfile).ReadAsArray()
        OpencvImg_data = np.zeros((Geodata.shape[1],Geodata.shape[2],Geodata.shape[0]))
        for i in range(Geodata.shape[0]):
            OpencvImg_data[:,:,i] = Geodata[Geodata.shape[0]-i-1,:,:]
        PredImg = np.rot90(OpencvImg_data, 2)
        im_height = PredImg.shape[0]                                                    # 列数
        im_width = PredImg.shape[1]                                                     # 行数
        print(im_width, im_height)
    else:
        PredImg = cv2.imread(imgfile) # print(type(im))
        im_height = PredImg.shape[0]
        im_width = PredImg.shape[1]
        print(im_width, im_height)
    return PredImg,im_height,im_width

if __name__ == '__main__':
    config_file = 'configs/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco.py'
    checkpoint_file = 'work_dirs/faster_rcnn_x101_32x4d_fpn_1x_coco/latest.pth'
    image_size = [448,448]
    imgfile = 'UAV3.jpg'
    SavePath = 'pred/'

    # build the model from a config file and a checkpoint file
    # model = init_detector(config_file, checkpoint_file, device='cuda:0')
    model = init_detector(config_file, checkpoint_file)
    PredImg,im_height,im_width = ImgRead(imgfile)
    bboxouts = SlideIndex(PredImg,im_height,im_width,model)

    correctbbox = Categorymerge(bboxouts)
    model.show_result(PredImg, correctbbox, out_file=SavePath + "gouziya.png")


    # for i in bboxouts:
    #     resultImg = visualize(PredImg,bboxouts)
    # cv2.imwrite(SavePath + "/gouzi.png",resultImg)

