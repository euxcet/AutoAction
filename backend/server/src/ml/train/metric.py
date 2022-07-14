from torch.nn import functional as F

def calc_accuracy(model, loader, device=None):
    ''' Use model to predict all data in the loader.
        Return the overall accuracy.
    args:
        model: child of nn.Module, the training model
        loader: torch.DataLoader
        device: the training device, None for 'cpu'
    return:
        float: the prediction accuracy
    '''
    correct, total = 0, 0
    for x_val, y_val in loader:
        # x_val.size(): [batch_size, length, channels]
        # v_val.size(): [batch_size]
        if device is not None: x_val, y_val = [t.to(device) for t in (x_val, y_val)]
        else: x_val, y_val = [t for t in (x_val, y_val)]
        out = model(x_val) # forward prediction
        preds = F.log_softmax(out, dim=1).argmax(dim=1) # predicted
        total += y_val.size(0) # total times: batch_size
        correct += (preds == y_val).sum().item() # correct times
    return float(correct) / float(total) # accuracy

def calc_metric(model, loader, device=None):
    ''' Calc some metric of the training model.
        Only implemented accuracy by now.
    args: the same as calc_accuracy()
    return: the same as calc_accuracy()
    '''
    return calc_accuracy(model, loader, device=device)