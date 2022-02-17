python -m onnxsim best.onnx best_sim.onnx
./build/tools/onnx/onnx2ncnn best_sim.onnx best.param best.bin
