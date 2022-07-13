import os
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from matplotlib import pyplot as plt

import file_utils
from ml.train.model import LSTMClassifier
from ml.train.dataset import create_train_val_loader
from ml.train.metric import calc_metric
from ml.train.export import export_pt, export_onnx, export_pth

class CyclicLR(_LRScheduler):
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]

def cosine(t_max, eta_min=0):
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * t / t_max)) / 2
    return scheduler


def train_model(trainId:str, timestamp:int, config:dict):
    # get input and output file paths
    ROOT = file_utils.get_train_path(trainId)
    X_TRAIN_PATH = os.path.join(ROOT, 'X_train.csv')    # train_data
    Y_TRAIN_PATH = os.path.join(ROOT, 'Y_train.csv')    # train_labels
    X_VAL_PATH = os.path.join(ROOT, 'X_val.csv')        # val_data
    Y_VAL_PATH = os.path.join(ROOT, 'Y_val.csv')        # val_labels
    X_TEST_PATH = os.path.join(ROOT, 'X_test.csv')      # test_data
    Y_TEST_PATH = os.path.join(ROOT, 'Y_test.csv')      # test_labels
    OUT_PATH_PTH = os.path.join(ROOT, 'best.pth')
    OUT_PATH_PT = os.path.join(ROOT, 'best.pt')
    OUT_PATH_ONNX = os.path.join(ROOT, 'best.onnx')

    # parse hyperparameters
    try:
        CONFIG_CHANNEL_DIM = config['channel_dim']
        CONFIG_SEQUENCE_DIM = config['sequence_dim']
        CONFIG_LAYER_DIM = config['layer_dim']
        CONFIG_HIDDEN_DIM = config['hidden_dim']
        CONFIG_OUTPUT_DIM = config['output_dim']
        CONFIG_LR = config['lr']
        CONFIG_EPOCH = config['epoch']
        CONFIG_USE_CUDA = config['use_cuda']
    except KeyError:
        return

    # config device
    if CONFIG_USE_CUDA:
        torch.cuda.set_device(0)
        
    # reset random seed
    np.random.seed(0)

    # create train and val data loader
    train_loader, val_loader = create_train_val_loader(X_TRAIN_PATH, Y_TRAIN_PATH)
    
    model = LSTMClassifier(CONFIG_CHANNEL_DIM, CONFIG_HIDDEN_DIM, CONFIG_LAYER_DIM, CONFIG_OUTPUT_DIM, use_cuda=CONFIG_USE_CUDA)
    if CONFIG_USE_CUDA:
        model = model.cuda()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=CONFIG_LR)
    scheduler = CyclicLR(optimizer, cosine(t_max=len(train_loader) * 2, eta_min=CONFIG_LR/100))
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    print('Start training')
    for epoch in range(0, CONFIG_EPOCH):
        for x_batch, y_batch in train_loader:
            model.train()
            if CONFIG_USE_CUDA:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
        model.eval()
        
        train_acc = calc_metric(model, train_loader, CONFIG_USE_CUDA)
        val_acc = calc_metric(model, val_loader, CONFIG_USE_CUDA)

        if epoch % 1 == 0:
            print(f'Epoch: {epoch:3d}, loss: {loss.item():.3f},' \
                f'train_acc: {train_acc:.3f}, val_acc: {val_acc:.3f}')

        if val_acc > best_acc:
            best_acc = val_acc
            export_pth(model, OUT_PATH_PTH)
            export_pt(model, OUT_PATH_PT, CONFIG_SEQUENCE_DIM, CONFIG_CHANNEL_DIM, CONFIG_USE_CUDA)
            export_onnx(model, OUT_PATH_ONNX, CONFIG_SEQUENCE_DIM, CONFIG_CHANNEL_DIM, CONFIG_USE_CUDA)
            print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
