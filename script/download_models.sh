mkdir tmp

# backend files
mkdir -p ../backend/server/data/file/
wget https://cloud.tsinghua.edu.cn/f/4c1bd41091f6443fb903/?dl=1 -O tmp/backend_file.zip
unzip -o tmp/backend_file.zip -d ../backend/server/data/file/

rm -r tmp
