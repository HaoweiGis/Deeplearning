#!/bin/bash
if [ $# != 5 ];then
echo "fname,folder,size,repeat,oname"
exit
fi

name=$1
fname=`echo $name|cut -d'.' -f1`
folder=$2
size=$3
repeat=$4
oname=$5


echo $repeat,$size
step=`echo | awk "{print $size-$size*$repeat}"`
step=`echo $step | cut -d. -f1`
echo 'step and size:'$step,$size

wi=`gdalinfo ${fname}.tif|grep ^Size|cut -d' ' -f3-4|sed -e 's/ //g'|cut -d',' -f1`
hi=`gdalinfo ${fname}.tif|grep ^Size|cut -d' ' -f3-4|sed -e 's/ //g'|cut -d',' -f2`

let x=${wi}/$size
let y=${hi}/$size

echo $x,$y
for i in `seq 0 ${x}`;
  do
  echo $i
    for j in `seq 0 ${y}`;
      do
      echo $j
      let ox=$size*$i;
      let oy=$size*$j;
      let xmax=$ox+$size;
      let ymax=$oy+$size;
       
      let ox1=$size*$i+$step;
      let oy1=$size*$j+$step;
      let xmax1=$ox1+$size;
      let ymax1=$oy1+$size;

      let ox2=$size*$i+$step;
      let oy2=$size*$j;
      let xmax2=$ox2+$size;
      let ymax2=$oy2+$size;

      let ox3=$size*$i;
      let oy3=$size*$j+$step;
      let xmax3=$ox3+$size;
      let ymax3=$oy3+$size;

      # echo $ox $oy $xmax $ymax $ox1 $oy1 $xmax1 $ymax1

      gdalwarp -overwrite -to SRC_METHOD=NO_GEOTRANSFORM -to DST_METHOD=NO_GEOTRANSFORM -te $ox $oy $xmax $ymax ${fname}.tif ${folder}/r_${oname}_${ox}_${oy}.tif
      gdal_translate -of PNG ${folder}/r_${oname}_${ox}_${oy}.tif ${folder}/r_${oname}_${ox}_${oy}.png

      # # image Flip
      gdalwarp -to SRC_METHOD=NO_GEOTRANSFORM -te $ox1 $oy1 $xmax1 $ymax1 ${fname}.tif ${folder}/r_${oname}_${ox1}_${oy1}.tif
      gdal_translate -of PNG ${folder}/r_${oname}_${ox1}_${oy1}.tif ${folder}/r_${oname}_${ox1}_${oy1}.png

      # repect enhance
      gdalwarp -overwrite -to SRC_METHOD=NO_GEOTRANSFORM -to DST_METHOD=NO_GEOTRANSFORM -te $ox2 $oy2 $xmax2 $ymax2 ${fname}.tif ${folder}/r_${oname}_${ox2}_${oy2}.tif
      gdal_translate -of PNG ${folder}/r_${oname}_${ox2}_${oy2}.tif ${folder}/r_${oname}_${ox2}_${oy2}.png

      # repect enhance
      gdalwarp -to SRC_METHOD=NO_GEOTRANSFORM -te $ox3 $oy3 $xmax3 $ymax3 ${fname}.tif ${folder}/r_${oname}_${ox3}_${oy3}.tif
      gdal_translate -of PNG ${folder}/r_${oname}_${ox3}_${oy3}.tif ${folder}/r_${oname}_${ox3}_${oy3}.png

    done
done
rm ${folder}/*.tif
rm ${folder}/*.png.aux.xml

