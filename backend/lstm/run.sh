rm -r output/
unzip output.zip
python main.py
python -m onnxsim test.onnx test_sim.onnx
onnx2ncnn test_sim.onnx test.param test.bin
