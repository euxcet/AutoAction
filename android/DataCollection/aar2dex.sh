cd contextactionlibrary/build/outputs/aar
rm -r contextactionlibrary-debug
mv contextactionlibrary-debug.aar contextactionlibrary-debug.zip
unzip contextactionlibrary-debug.zip -d contextactionlibrary-debug
cp contextactionlibrary-debug/classes.jar ../../../../../../backend/server/data/file/
cd ../../../../../../backend/server/data/file/
# TODO: configurate d8 path
~/Library/Android/sdk/build-tools/32.0.0/d8 classes.jar

