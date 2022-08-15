import torch
import os
import subprocess

def export_pth(model, PTH_PATH):
    torch.save(model.state_dict(), PTH_PATH)

def export_pt(model, PT_PATH, sequence_dim, channel_dim, device=None):
    if device is not None:
        libtorch_model = torch.jit.trace(model,
            torch.rand(1, sequence_dim, channel_dim).to(device))
    else:
        libtorch_model = torch.jit.trace(model,
            torch.rand(1, sequence_dim, channel_dim))
    libtorch_model.save(PT_PATH)

def export_onnx(model, ONNX_PATH, sequence_dim, channel_dim, device=None):
    if device is not None:
        input_x = torch.randn(1, sequence_dim, channel_dim).to(device)
    else:
        input_x = torch.randn(1, sequence_dim, channel_dim)
    torch.onnx.export(model, input_x, ONNX_PATH, opset_version=10,
        do_constant_folding=True, keep_initializers_as_inputs=True,
        verbose=False, input_names=["input"], output_names=["output"])

def export_mnn(ONNX_PATH, MNN_PATH):
    converter = os.getenv("MNN_CONVERTER")
    print(ONNX_PATH)
    print(MNN_PATH)
    cmd = converter + " -f ONNX --modelFile " + ONNX_PATH + " --MNNModel " + MNN_PATH + " --bizCode biz"
    try:
        os.system(cmd)
        # subprocess.call(cmd, shell=True, executable="/bin/zsh")
    except Exception as e:
        print("Error occurs while exporting the mnn model: %s" % e)