#bin/bash

mkdir train test/ val
mkdir train/image test/image val/image
mkdir train/label test/label val/label

while read line;do cp image/$line.png train/image/${line}.png;done < index/train.txt
while read line;do cp image/$line.png test/image/${line}.png;done < index/test.txt
while read line;do cp image/$line.png val/image/${line}.png;done < index/val.txt
ls train/image/ |wc -l
ls test/image/ |wc -l
ls val/image/ |wc -l

while read line;do cp label/$line.png train/label/${line}_building.png;done < index/train.txt
while read line;do cp label/$line.png test/label/${line}_building.png;done < index/test.txt
while read line;do cp label/$line.png val/label/${line}_building.png;done < index/val.txt
ls train/label/ |wc -l
ls test/label/ |wc -l
ls val/label/ |wc -l

# rm -r image index label