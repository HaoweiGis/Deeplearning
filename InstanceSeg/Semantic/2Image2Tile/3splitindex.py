import random
import sys

nums = int(sys.argv[1])
num = int(sys.argv[2])
num1 = nums-num
num2 = num1-num
print(num1,num2)
with open('all.txt','r')as f:
    lines = f.readlines()
    g = [i for i in range(nums)]# 设置文件总数
    random.shuffle(g)
    # 设置需要的文件数
    train = g[:num2]
    trainval = g[num2:num1]
    val = g[num1:]

    for index, line in enumerate(lines,0):
        if index in train:
            with open('train.txt','a')as trainf:
                trainf.write(line)
        elif index in trainval:
            with open('test.txt','a')as trainvalf:
                trainvalf.write(line)
        elif index in val:
            with open('val.txt','a')as valf:
                valf.write(line)
