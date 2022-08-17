cd ContextActionLibrary/build/outputs/aar
rm -r contextactionlibrary-debug
mv contextactionlibrary-debug.aar contextactionlibrary-debug.zip
unzip contextactionlibrary-debug.zip -d contextactionlibrary-debug
mkdir -p ../../../../../../backend/server/data/file/
cp contextactionlibrary-debug/classes.jar ../../../../../../backend/server/data/file/
cd ../../../../../../backend/server/data/file/
# TODO: configurate d8 path
$D8_PATH classes.jar
mv classes.dex release.dex
curl -XPOST http://114.67.110.181:6125/file -F "file=@release.dex"
curl -XPOST http://114.67.110.181:6125/md5 
