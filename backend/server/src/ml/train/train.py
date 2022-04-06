import os
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

from ml.train.model import LSTMClassifier
#from ml.train.dataset import create_datasets, create_loaders
from ml.train.dataset import create_train_val_loader
from ml.train.metric import calculate_metrics
from ml.train.export import export_pt, export_onnx, export_pth
import file_utils

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


def train_model(trainId, timestamp, config):
    ROOT = file_utils.get_train_path(trainId)
    TRAIN_X_PATH = os.path.join(ROOT, 'X_train.csv')
    TRAIN_Y_PATH = os.path.join(ROOT, 'Y_train.csv')
    VALIDATION_X_PATH = os.path.join(ROOT, 'X_val.csv')
    VALIDATION_Y_PATH = os.path.join(ROOT, 'Y_val.csv')
    TEST_X_PATH = os.path.join(ROOT, 'X_test.csv')
    TEST_Y_PATH = os.path.join(ROOT, 'Y_test.csv')

    OUTPUT_PTH_PATH = os.path.join(ROOT, 'best.pth')
    OUTPUT_PT_PATH = os.path.join(ROOT, 'best.pt')
    OUTPUT_ONNX_PATH = os.path.join(ROOT, 'best.onnx')

    try:
        '''
            CHANNEL_DIM 6
            SEQUENCE_DIM 128
            LAYER_DIM 1
            HIDDEN_DIM 512
            OUTPUT_DIM 2
            LR 0.001
            EPOCH 200
            USE_CUDA True
        '''
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

    if CONFIG_USE_CUDA:
        torch.cuda.set_device(0)
    np.random.seed(0)

    train_loader, val_loader = create_train_val_loader(TRAIN_X_PATH, TRAIN_Y_PATH)
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

        print('train:', calculate_metrics(model, train_loader, CONFIG_USE_CUDA))
        val_acc = calculate_metrics(model, val_loader, CONFIG_USE_CUDA)
        print('val:', val_acc)

        if epoch % 5 == 0:
            print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Val Acc.: {val_acc:2.2%}')

        if val_acc > best_acc:
            best_acc = val_acc
            export_pth(model, OUTPUT_PTH_PATH)
            export_pt(model, OUTPUT_PT_PATH, CONFIG_SEQUENCE_DIM, CONFIG_CHANNEL_DIM, CONFIG_USE_CUDA)
            export_onnx(model, OUTPUT_ONNX_PATH, CONFIG_SEQUENCE_DIM, CONFIG_CHANNEL_DIM, CONFIG_USE_CUDA)
            print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')


if __name__ == '__main__':
    pass
    # train('9087654321', False)