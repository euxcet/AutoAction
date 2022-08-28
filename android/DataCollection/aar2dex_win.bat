@echo off
pushd .
set server_addr=127.0.0.1:6125

chdir ContextActionLibrary\build\outputs\aar
rmdir /S /Q contextactionlibrary-debug
move /Y contextactionlibrary-debug.aar contextactionlibrary-debug.zip
mkdir contextactionlibrary-debug
tar -xf contextactionlibrary-debug.zip -C contextactionlibrary-debug
mkdir ..\..\..\..\..\..\backend\server\data\file\
copy /Y contextactionlibrary-debug\classes.jar ..\..\..\..\..\..\backend\server\data\file\
chdir ..\..\..\..\..\..\backend\server\data\file\

rem TODO: configurate d8 path
rem set sdk_version=30.0.3
rem set D8_PATH=%USERPROFILE%\AppData\Local\Android\Sdk\build-tools\%sdk_version%\d8.bat
call "%D8_PATH%" classes.jar
if errorlevel 1 goto err
move classes.dex release.dex
adb push release.dex /sdcard/Android/data/com.hcifuture.scanner/files/context-lib

:checkarg
if [%~1]==[-s] if not [%~2]==[] set server_addr=%~2
goto updatefile

:updatefile
echo Updating server at %server_addr%
curl -XPOST http://%server_addr%/file -F "file=@release.dex"
if errorlevel 1 goto err else goto updatemd5

:updatemd5
curl -XPOST http://%server_addr%/md5
if errorlevel 1 goto err else goto noerr

:noerr
echo Exiting with no error.
goto end

:err
echo Exiting with error!
goto end

:end
popd
