from torch.nn import functional as F
import numpy as np
from sklearn import metrics

def calc_metric(model, loader, class_num, device=None):
    ''' Use model to predict all data in the loader.
        Return metrics.
    args:
        model: child of nn.Module, the training model
        loader: torch.DataLoader
        device: the training device, None for 'cpu'
        class_num: the number of classes
    return:
        metrics: dict
    '''
    y_pred = np.zeros(0)
    y_true = np.zeros(0)

    for x_val, y_val in loader:
        # x_val.size(): [batch_size, length, channels]
        # y_val.size(): [batch_size]

        if device is not None: x_val, y_val = [t.to(device) for t in (x_val, y_val)]
        else: x_val, y_val = [t for t in (x_val, y_val)]
        out = model(x_val) # forward prediction
        preds = F.log_softmax(out, dim=1).argmax(dim=1) # predicted
        y_pred = np.append(y_pred, preds.cpu().numpy())
        y_true = np.append(y_true, y_val.cpu().numpy())

    accuracy = metrics.accuracy_score(y_true, y_pred)
    balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred, average="weighted")
    precision = metrics.precision_score(y_true, y_pred, average="weighted")
    f1_score = metrics.fbeta_score(y_true, y_pred, beta=1, average="weighted")
    f0_5_score = metrics.fbeta_score(y_true, y_pred, beta=0.5, average="weighted")
    f2_score = metrics.fbeta_score(y_true, y_pred, beta=2, average="weighted")

    if class_num > 2:
        auc = 0.0
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        auc = metrics.auc(fpr, tpr)

    '''
    print(mat)
    recall = 0
    recall_count = [0 for i in range(class_num)]
    precision = 0
    precision_count = [0 for i in range(class_num)]

    for i in range(class_num):
        for j in range(class_num):
            recall_count[i] += mat[j][i]
            precision_count[i] += mat[i][j]

    for i in range(class_num):
        recall += mat[i][i] / recall_count[i]
        precision += mat[i][i] / precision_count[i]

    recall /= class_num
    precision /= class_num
    beta = 0.5
    f1 = (1 + beta * beta) * (precision * recall) / (beta * beta * precision + recall)
    '''

    metrics_result = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'recall': recall,
        'precision': precision,
        'f1_score': f1_score,
        'f0_5_score': f0_5_score,
        'f2_score': f2_score,
        'auc': auc
    }

    return metrics_result