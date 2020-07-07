#!/bin/bash
# bash clip_train_image.sh /data/datasets/Image_training/lanzhou_img_RGB.tif ../allimage/ 513 lanzhou
# bash clip_train_label.sh /data/datasets/Image_training/lanzhou_building_footprints.tif ../alllabel/ 513 lanzhou

# image  index  label
workdir=/data/InstanceSeg/Semantic/ImageDataDP
mkdir ${workdir}/image ${workdir}/index ${workdir}/label

bash clip_train_image.sh ${workdir}/tile_L51E024022sampleI.tif ${workdir}/image/ 512 0.5 Subset1
bash clip_train_image.sh ${workdir}/tile_L51E024023sampleI.tif ${workdir}/image/ 512 0.5 Subset2
# bash clip_train_image.sh ${workdir}/lanzhou3sampleI.tif ${workdir}/image/ 512 0.5 Subset3
# bash clip_train_image.sh ${workdir}/xian1sampleI.tif ${workdir}/image/ 512 0.5 Subset4
# bash clip_train_image.sh ${workdir}/xian2sampleI.tif ${workdir}/image/ 512 0.5 Subset5

bash clip_train_label.sh ${workdir}/tile_L51E024022sampleL.tif ${workdir}/label/ 512 0.5 Subset1
bash clip_train_label.sh ${workdir}/tile_L51E024023sampleL.tif ${workdir}/label/ 512 0.5 Subset2
# bash clip_train_label.sh ${workdir}/lanzhou3sampleL.tif ${workdir}/label/ 512 0.5 Subset3
# bash clip_train_label.sh ${workdir}/xian1sampleL.tif ${workdir}/label/ 512 0.5 Subset4
# # bash clip_train_label.sh ${workdir}/xian2sampleL.tif ${workdir}/label/ 512 0.5 Subset5

ls ${workdir}/image/*.png |wc -l
ls ${workdir}/label/*.png |wc -l