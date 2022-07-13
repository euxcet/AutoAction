import torch

def export_pth(model, PTH_PATH):
    torch.save(model.state_dict(), PTH_PATH)

def export_pt(model, PT_PATH, sequence_dim, channel_dim, use_cuda):
    if use_cuda:
        libtorch_model = torch.jit.trace(model,
            torch.rand(1, sequence_dim, channel_dim).cuda())
    else:
        libtorch_model = torch.jit.trace(model,
            torch.rand(1, sequence_dim, channel_dim))
    libtorch_model.save(PT_PATH)

def export_onnx(model, ONNX_PATH, sequence_dim, channel_dim, use_cuda):
    if use_cuda:
        input_x = torch.randn(1, sequence_dim, channel_dim).cuda()
    else:
        input_x = torch.randn(1, sequence_dim, channel_dim)
    torch.onnx.export(model, input_x, ONNX_PATH, opset_version=10,
        do_constant_folding=True, keep_initializers_as_inputs=True,
        verbose=False, input_names=["input"], output_names=["output"])