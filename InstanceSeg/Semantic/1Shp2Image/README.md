## 文件使用
！shp文件里面需要包含一个字段，class，类别计算从1开始，类别类型为短整型  坐标系必须是3857
！tif文件放在Samples里面，后缀为sample.tif 坐标系必须是3857

python shp2img.py `pwd` --shp-name filename.shp  
会经过三个步骤:1shp转geoTiff; 2img转byte; 3img clip img  
最终就会得到影像和标签数据：存放在Samples里面后缀分别为I.tif，L.tif
mv *I.tif *L.tif targetPath

### 过程记录

生成单类别的影像-`gdal_rasterize -a class -where 'class=1' -l hulunbuir3857 -tr 10 10 -init 0 -a_nodata 0 hulunbuir3857.shp output1.tif`  

生成包含全类别的影像-`gdal_rasterize -a class -l hulunbuir3857 -tr 10 10 -init 0 -a_nodata 0 hulunbuir3857.shp output1.tif`  

参考：https://gdal.org/programs/gdal_rasterize.html#cmdoption-gdal-translate-ts

16位转byte:gdal_translate -of GTiff -ot Byte -scale 0 65535 0 255 src_dataset dst_dataset(代码中进行了拉伸优化)

多波段影像需要转换成RGB格式
gdal_translate  -b 3 -b 2 -b 1 source target
转换为3857：gdalwarp -t_srs EPSG:3857 input.tif output.tif