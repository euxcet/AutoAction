import os
from multiprocessing import cpu_count
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from process.train.model import LSTMClassifier
import fileUtils


def create_datasets(X, y, test_size=0.2, drop_cols=None, time_dim_first=False):
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)
    X_grouped = create_grouped_array(X, drop_cols=drop_cols)
    if time_dim_first:
        X_grouped = X_grouped.transpose(0, 2, 1)
    X_train, X_valid, y_train, y_valid = train_test_split(X_grouped, y_enc, test_size=0.1)
    X_train, X_valid = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_valid)]
    y_train, y_valid = [torch.tensor(arr, dtype=torch.long) for arr in (y_train, y_valid)]
    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    return train_ds, valid_ds, enc


def create_grouped_array(data, group_col='series_id', drop_cols=None):
    X_grouped = np.row_stack([
        group.drop(columns=drop_cols).values[None]
        for _, group in data.groupby(group_col)])
    return X_grouped


def create_test_dataset(X, drop_cols=None):
    X_grouped = np.row_stack([
        group.drop(columns=drop_cols).values[None]
        for _, group in X.groupby('series_id')])
    X_grouped = torch.tensor(X_grouped.transpose(0, 2, 1)).float()
    y_fake = torch.tensor([0] * len(X_grouped)).long()
    return TensorDataset(X_grouped, y_fake)


def create_loaders(train_ds, valid_ds, bs=512, jobs=0):
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, bs, shuffle=False, num_workers=jobs)
    return train_dl, valid_dl


def accuracy(output, target):
    return (output.argmax(dim=1) == target).float().mean().item()



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
        return eta_min + (base_lr - eta_min)*(1 + np.cos(np.pi*t/t_max))/2
    return scheduler


def train_model(timestamp, use_cuda):
    if use_cuda:
        torch.cuda.set_device(0)
    np.random.seed(1)

    ROOT = fileUtils.get_train_path(timestamp)
    print(ROOT)
    TRAIN = os.path.join(ROOT, 'X_train.csv')
    TARGET = os.path.join(ROOT, 'y_train.csv')
    TEST = os.path.join(ROOT, 'X_test.csv')
    PTH_PATH = os.path.join(ROOT, 'best.pth')
    PT_PATH = os.path.join(ROOT, 'best.pt')
    ONNX_PATH = os.path.join(ROOT, 'best.onnx')

    ID_COLS = ['series_id', 'measurement_number']

    print(ROOT, TRAIN, TARGET, TEST)

    x_cols = {
        'series_id': np.uint32,
        'measurement_number': np.uint32,
        'angular_velocity_X': np.float32,
        'angular_velocity_Y': np.float32,
        'angular_velocity_Z': np.float32,
        'linear_acceleration_X': np.float32,
        'linear_acceleration_Y': np.float32,
        'linear_acceleration_Z': np.float32
    }

    y_cols = {
        'series_id': np.uint32,
        'group_id': np.uint32,
        'group_name': str
    }

    x_trn = pd.read_csv(TRAIN, usecols=x_cols.keys(), dtype=x_cols)
    x_tst = pd.read_csv(TEST, usecols=x_cols.keys(), dtype=x_cols)
    y_trn = pd.read_csv(TARGET, usecols=y_cols.keys(), dtype=y_cols)

    n = 100
    sched = cosine(n)
    lrs = [sched(t, 1) for t in range(n * 4)]

    print('Preparing datasets')
    trn_ds, val_ds, enc = create_datasets(x_trn, y_trn['group_name'], drop_cols=ID_COLS)


    bs = 32
    print(f'Creating data loaders with batch size: {bs}')
    trn_dl, val_dl = create_loaders(trn_ds, val_ds, bs, jobs=cpu_count())

    input_dim = 6
    hidden_dim = 512
    layer_dim = 2
    output_dim = 5
    seq_dim = 128

    lr = 0.001
    n_epochs = 200
    iterations_per_epoch = len(trn_dl)
    best_acc = 0
    patience, trials = 100, 0

    model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim, use_cuda=False)
    if use_cuda:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    sched = CyclicLR(opt, cosine(t_max=iterations_per_epoch * 2, eta_min=lr/100))

    print('Start model training')
    for epoch in range(1, n_epochs + 1):
        
        for i, (x_batch, y_batch) in enumerate(trn_dl):
            model.train()
            if use_cuda:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            sched.step()
            opt.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            opt.step()
        
        model.eval()

        correct, total = 0, 0
        for x_val, y_val in trn_dl:
            if use_cuda:
                x_val, y_val = [t.cuda() for t in (x_val, y_val)]
            else:
                x_val, y_val = [t for t in (x_val, y_val)]
            out = model(x_val)
            preds = F.log_softmax(out, dim=1).argmax(dim=1)
            total += y_val.size(0)
            correct += (preds == y_val).sum().item()
        print('train:', correct / total)

        correct, total = 0, 0
        for x_val, y_val in val_dl:
            if use_cuda:
                x_val, y_val = [t.cuda() for t in (x_val, y_val)]
            else:
                x_val, y_val = [t for t in (x_val, y_val)]
            out = model(x_val)
            preds = F.log_softmax(out, dim=1).argmax(dim=1)
            total += y_val.size(0)
            correct += (preds == y_val).sum().item()

        
        acc = correct / total
        print('val:', acc)

        if epoch % 5 == 0:
            print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%}')

        if acc > best_acc:
            trials = 0
            best_acc = acc
            torch.save(model.state_dict(), PTH_PATH)

            if use_cuda:
                libtorch_model = torch.jit.trace(model, torch.rand(1, 128, 6).cuda())
            else:
                libtorch_model = torch.jit.trace(model, torch.rand(1, 128, 6))
            libtorch_model.save(PT_PATH)

            if use_cuda:
                input_x = torch.randn(1, 128, 6).cuda()
            else:
                input_x = torch.randn(1, 128, 6)
            torch.onnx.export(model, input_x, ONNX_PATH, opset_version=10, do_constant_folding=True, keep_initializers_as_inputs=True, verbose=False, input_names=["input"], output_names=["output"])

            print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
        else:
            trials += 1


if __name__ == '__main__':
    train('9087654321', False)