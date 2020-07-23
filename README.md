# Deeplearning
 mmdetection2，detectron2学习记录


for i in *.tif;do name=`ls $i|cut -d. -f1`; gdal_translate $i ../image/${name}.jpg ;done