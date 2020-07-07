## 文件使用
1. bash 1clip_sampleimg.sh  修改对应文件路径和文件名，会在对应工作目录下面建立Image,label,index文件夹，用来存放Tile.

2. 到影像Image路径下面ls *.png |cut -d. -f1>../all.txt

3. python 2splitindex.py workpath -a xxx -v xxx  (ls *.png|cut -d. -f1 >../all.txt)  
work_path工作路径，--allnum表示tile文件总数，-v表示验证集和测试集文件数量。

### 过程记录

bash clip_train_image.sh ${workdir}/SampleImg/Imagename.tif ${workdir}/image/ size repeat Subsetx

image_enhance.py 图像增强（后续可以不断补充）