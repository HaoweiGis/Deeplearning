## 文件使用
1. 1Shp2Image  将shp文章转换为与遥感影像尺寸一样的大小，输出存储在Samples *I.tif *L.tif  
2. 2Image2Tile 将影像切片，保存在image，label,index中  
3. refile.sh  将数据image，label,index 转换成train test/ val，方便转换成CoCo

## 记录
gdal_calc.py -A GF2_PMS1__L1A0001064454-MSS1_label.tif --A_band=1 -B GF2_PMS1__L1A0001064454-MSS1_label.tif --B_band=2 -C GF2_PMS1__L1A0001064454-MSS1_label.tif --C_band=3 --outfile output.tif --calc="(A==200)*(B==0)*(C==0)*1 + (A==250)*(B==0)*(C==150)*2 + (A==200)*(B==150)*(C==150)*3 + (A==250)*(B==150)*(C==150)*4 + (A==0)*(B==200)*(C==0)*5 + (A==150)*(B==250)*(C==0)*6 + (A==150)*(B==200)*(C==150)*7 + (A==200)*(B==0)*(C==200)*8 + (A==150)*(B==0)*(C==250)*9 + (A==150)*(B==150)*(C==250)*10 + (A==250)*(B==200)*(C==0)*11 + (A==200)*(B==200)*(C==0)*12 + (A==0)*(B==0)*(C==200)*13 + (A==0)*(B==150)*(C==200)*14 + (A==0)*(B==200)*(C==250)*15"

for i in *.tif;do name=`echo $i|cut -d. -f1`;gdal_calc.py -A $i --A_band=1 -B $i --B_band=2 -C $i --C_band=3 --outfile ${name}_calc.tif --calc="(A==200)*(B==0)*(C==0)*1 + (A==250)*(B==0)*(C==150)*2 + (A==200)*(B==150)*(C==150)*3 + (A==250)*(B==150)*(C==150)*4 + (A==0)*(B==200)*(C==0)*5 + (A==150)*(B==250)*(C==0)*6 + (A==150)*(B==200)*(C==150)*7 + (A==200)*(B==0)*(C==200)*8 + (A==150)*(B==0)*(C==250)*9 + (A==150)*(B==150)*(C==250)*10 + (A==250)*(B==200)*(C==0)*11 + (A==200)*(B==200)*(C==0)*12 + (A==0)*(B==0)*(C==200)*13 + (A==0)*(B==150)*(C==200)*14 + (A==0)*(B==200)*(C==250)*15";done


for i in *calc.tif;do gdal_edit.py $i -unsetnodata;done
for i in *_calc.tif;do line=`gdalinfo -hist $i|grep -A 1 '256 buckets from'`;echo $line;done
