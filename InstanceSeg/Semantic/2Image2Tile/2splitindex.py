import random
import sys
import os 
import argparse



def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('work_path', help='work path')
    parser.add_argument('--allnum','-a', help='shp data path')
    parser.add_argument('--valnum','-v', help='raster data path', default='Samples')
    args = parser.parse_args()
    return args


args = parse_args()
workpath = args.work_path
nums = int(args.allnum)
num = int(args.valnum)
# nums = int(sys.argv[1])
# num = int(sys.argv[2])
# workpath=sys.argv[3]
num1 = nums-num
num2 = num1-num
allfile = os.path.join(workpath,'all.txt')
trainfile =os.path.join(workpath,'index/train.txt')
testfile =os.path.join(workpath,'index/test.txt')
valfile =os.path.join(workpath,'index/val.txt')
print(num1,num2)

with open(allfile,'r')as f:
    lines = f.readlines()
    g = [i for i in range(1, nums)]# 设置文件总数
    random.shuffle(g)
    # 设置需要的文件数
    train = g[:num2]
    trainval = g[num2:num1]
    val = g[num1-1:]
    print(len(train),len(trainval),len(val))

    for index, line in enumerate(lines,1):
        if index in train:
            with open(trainfile,'a')as trainf:
                trainf.write(line)
        elif index in val:
            with open(valfile,'a')as valf:
                valf.write(line)
        elif index in trainval:
            with open(testfile,'a')as trainvalf:
                trainvalf.write(line)
