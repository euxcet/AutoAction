from torch.nn import functional as F

# TODO
def calculate_accuracy(model, loader, use_cuda):
    correct, total = 0, 0
    for x_val, y_val in loader:
        if use_cuda:
            x_val, y_val = [t.cuda() for t in (x_val, y_val)]
        else:
            x_val, y_val = [t for t in (x_val, y_val)]
        out = model(x_val)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_val.size(0)
        correct += (preds == y_val).sum().item()
    return float(correct) / float(total)

def calculate_metrics(model, loader, use_cuda):
    return calculate_accuracy(model, loader, use_cuda)