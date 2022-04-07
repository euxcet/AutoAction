pwd
cd ../data/train/$1
python3 -m onnxsim best.onnx best-sim.onnx
../../../tool/onnx2ncnn best-sim.onnx best.param best.bin
cp best.pth ../../file/best.pth
cp best.bin ../../file/best.bin
cp best.param ../../file/best.param
