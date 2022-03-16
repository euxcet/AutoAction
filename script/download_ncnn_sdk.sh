mkdir tmp

# ncnn
wget https://cloud.tsinghua.edu.cn/f/eb29e537ac764c148fe7/?dl=1 -O tmp/ncnn.zip
unzip -o tmp/ncnn.zip -d ../android/DataCollection/NcnnLibrary/src/main/

# opencv
wget https://cloud.tsinghua.edu.cn/f/91275b6dd5f3447ba46e/?dl=1 -O tmp/opencv.zip
unzip -o tmp/opencv.zip -d ../android/DataCollection/NcnnLibrary/src/

rm -r tmp
