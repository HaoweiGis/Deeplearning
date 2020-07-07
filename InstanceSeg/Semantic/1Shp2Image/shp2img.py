from osgeo import gdal
from gdalconst import GA_ReadOnly
import os
import os.path as osp
import argparse
import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('work_path', help='work path')
    parser.add_argument('--shp-name', help='shp data path')
    parser.add_argument('--img-dir', help='raster data path', default='Samples')
    parser.add_argument('-o', '--out-dir', help='output path')
    args = parser.parse_args()
    return args


def collect_files(img_dir):
    suffix = 'sample.tif'
    files = []
    for img_file in glob.glob(os.path.join(img_dir, '*sample.tif')):
        assert img_file.endswith(suffix), img_file
        files.append(img_file)
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')
    return files

def img_stats(geofile,band):
    '''
    统计影像最大最小值方便图像拉伸
    '''
    srcband = geofile.GetRasterBand(band)
    # Get raster statistics
    stats = srcband.GetStatistics(True, True)
    return stats[0],stats[1]

def img2byte(geofile):
    # open raster and choose band to find min, max
    gtif = gdal.Open(geofile)
    min1,max1= img_stats(gtif,1)
    min2,max2= img_stats(gtif,2)
    min3,max3= img_stats(gtif,3)
    minV = min(min1,min2,min3)
    maxV = max(max1,max2,max3)
    os.system('gdal_translate -of GTiff -ot Byte -scale ' + ' '.join([str(x) for x in [minV, maxV, 0, 255]]) + ' -of GTiff '+ geofile + ' ' + geofile.replace('.tif','I.tif'))
    print(geofile + '  OK!')

def rasterCraster(geofile):
    # 裁剪
    data = gdal.Open(geofile, GA_ReadOnly)
    geoTransform = data.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * data.RasterXSize
    miny = maxy + geoTransform[5] * data.RasterYSize
    print('gdal_translate -ot Byte -projwin ' + ' '.join([str(x) for x in [minx, maxy, maxx, miny]]) + ' -of GTiff '+ shpfile.replace('.shp','.tif') + ' ' + geofile.replace('.tif','L.tif'))
    os.system('gdal_translate -ot Byte -projwin ' + ' '.join([str(x) for x in [minx, maxy, maxx, miny]]) + ' -of GTiff '+ shpfile.replace('.shp','.tif') + ' ' + geofile.replace('.tif','L.tif'))


# python gdal2byte.py --shp-name hulunbuir3857.shp
if __name__ == "__main__":
    args = parse_args()
    work_path = args.work_path
    shpfile = args.shp_name
    img_dir = osp.join(work_path,args.img_dir)
    imgfiles = collect_files(img_dir)

    # shp转geoTiff
    res = gdal.Open(imgfiles[0]).GetGeoTransform()[1]
    os.system('gdal_rasterize -ot Byte -a class -where "class=1" -tr ' + ' '.join([str(x) for x in [res, res]]) + ' -init 0 -a_nodata 0 ' + shpfile +' '+ shpfile.replace('.shp','.tif'))
    print("完成shp转geoTiff")

    # img转byte
    for img in imgfiles:
        img2byte(img)
    print("完成img转byte")

    # img clip img
    for img in imgfiles:
        rasterCraster(img)
    print("完成img clip img")

