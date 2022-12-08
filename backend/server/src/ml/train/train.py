import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from matplotlib import pyplot as plt

import file_utils
from ml.train.model import LSTMClassifier, CNNClassifier
from ml.train.dataset import create_train_val_loader
from ml.train.metric import calc_metric
from ml.train.export import export_pt, export_onnx, export_pth, export_mnn
from ml.global_vars import GlobalVars

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

def prepare_log_files(LOG_FOLDER):
    file_utils.mkdir(LOG_FOLDER)
    log_metrics_file = None
    for i in range(np.iinfo(np.int32).max):
        run_folder = os.path.join(LOG_FOLDER, 'run' + str(i))
        if not os.path.exists(run_folder):
            file_utils.mkdir(run_folder)
            log_metrics_file = os.path.join(run_folder, 'metrics.txt')
            break
    return log_metrics_file

def log_metrics(log_metrics_file:str, epoch:int, train_metrics:dict, valid_metrics:dict):
    with open(log_metrics_file, 'a') as fout:
        metrics_name = ['accuracy', 'balanced_accuracy', 'recall', 'precision', 'f1_score', 'f0_5_score', 'f2_score', 'auc']
        print("-----------------------------------------")
        fout.write("-----------------------------------------\n")
        print(f"Epoch: {epoch}")
        fout.write(f"Epoch: {epoch}\n")

        print("Metrics on train dataset:")
        fout.write("Metrics on train dataset:\n")
        for name in metrics_name:
            metric = train_metrics[name]
            print("\t{}: {:.3}".format(name, metric), end='')
            fout.write("\t{}: {:.3}".format(name, metric))
        print()
        fout.write('\n')
            
        print("Metrics on valid dataset:")
        fout.write("Metrics on valid dataset:\n")
        for name in metrics_name:
            metric = valid_metrics[name]
            print("\t{}: {:.3}".format(name, metric), end='')
            fout.write("\t{}: {:.3}".format(name, metric))
        print()
        fout.write('\n')

def train_model(trainId:str, timestamp:int, sessionId:int, config:dict):
    # get input and output file paths
    ROOT = file_utils.get_train_path(trainId)
    X_TRAIN_PATH = os.path.join(ROOT, 'X_train.csv')    # train_data
    Y_TRAIN_PATH = os.path.join(ROOT, 'Y_train.csv')    # train_labels
    X_TEST_PATH = os.path.join(ROOT, 'X_test.csv')      # test_data
    Y_TEST_PATH = os.path.join(ROOT, 'Y_test.csv')      # test_labels
    OUT_PATH_PTH = os.path.join(ROOT, 'best.pth')
    OUT_PATH_PT = os.path.join(ROOT, 'best.pt')
    OUT_PATH_ONNX = os.path.join(ROOT, 'best.onnx')
    OUT_PATH_MNN = os.path.join(ROOT, 'best.mnn')
    OUT_SESSION_PATH_PTH = os.path.join(ROOT, 'best_s' + str(sessionId) + '.pth')
    OUT_SESSION_PATH_PT = os.path.join(ROOT, 'best_s' + str(sessionId) + '.pt')
    OUT_SESSION_PATH_ONNX = os.path.join(ROOT, 'best_s' + str(sessionId) + '.onnx')
    OUT_SESSION_PATH_MNN = os.path.join(ROOT, 'best_s' + str(sessionId) + '.mnn')

    LOG_FOLDER = os.path.join(ROOT, 'log')
    log_metrics_file = prepare_log_files(LOG_FOLDER)

    device:str = GlobalVars.DEVICE
    if device == 'cuda' and torch.cuda.is_available():
        print(f'### Training device: cuda.')
        device = torch.device('cuda')
    elif device == 'mps' and torch.backends.mps.is_available():
        print(f'### Training device: mps.')
        device = torch.device('mps')
    else:
        print(f'### Training device: cpu.')
        device = None

    # parse generic hyperparameters 
    try:
        CONFIG_CHANNEL_DIM = config['channel_dim']
        CONFIG_SEQUENCE_DIM = config['sequence_dim']
        CONFIG_OUTPUT_DIM = config['output_dim']
        CONFIG_LR = config['lr']
        CONFIG_EPOCH = config['epoch']
    except KeyError:
        return

    # create the network
    backbone = GlobalVars.NETWORK_BACKBONE
    if backbone == 'lstm':
        # parse hyperparameters for lstm
        try:
            CONFIG_LAYER_DIM = config['lstm_layer_dim']
            CONFIG_HIDDEN_DIM = config['lstm_hidden_dim']
            CONFIG_FC_DIM = config['lstm_fc_dim']
        except KeyError:
            return
        model = LSTMClassifier(CONFIG_CHANNEL_DIM, CONFIG_HIDDEN_DIM, CONFIG_LAYER_DIM,
            CONFIG_FC_DIM, CONFIG_OUTPUT_DIM, device=device)
    elif backbone == 'cnn':
        # parse hyperparameters for cnn
        model = CNNClassifier(CONFIG_OUTPUT_DIM)

    if device is not None:
        model = model.to(device)
        
    # reset random seed
    np.random.seed(0)

    # create train and val data loader
    label_path = file_utils.get_train_label_path(trainId)
    train_loader, val_loader = create_train_val_loader(X_TRAIN_PATH, Y_TRAIN_PATH, label_path)
    
    # model = LSTMClassifier(CONFIG_CHANNEL_DIM, CONFIG_HIDDEN_DIM, CONFIG_LAYER_DIM,
    #     CONFIG_FC_DIM, CONFIG_OUTPUT_DIM, device=device)
    if device is not None:
        model = model.to(device)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=CONFIG_LR)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG_LR)
#scheduler = CyclicLR(optimizer, cosine(t_max=len(train_loader) * 2, eta_min=CONFIG_LR/100))
    criterion = nn.CrossEntropyLoss()


    # save configs to the log file.
    print(GlobalVars.__dict__)
    with open(log_metrics_file, 'a') as fout:
        fout.write("Config:\n")
        for k, v in config.items():
            fout.write(k + " " + str(v) + "\n")

        fout.write("\n\n")

        fout.write("Global Variables:\n")
        for k, v in GlobalVars.__dict__.items():
            if not k.startswith('_'):
                fout.write(k + " " + str(v) + "\n")
        fout.write("\n\n")


    best_metric = 0.0

    print('Start training!')
    for epoch in range(0, CONFIG_EPOCH):
        model.train()
        for x_batch, y_batch in train_loader:
            if device is not None:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

        
        model.eval()
        train_metrics = calc_metric(model, train_loader, CONFIG_OUTPUT_DIM, device=device)
        valid_metrics = calc_metric(model, val_loader, CONFIG_OUTPUT_DIM, device=device)

        log_metrics(log_metrics_file, epoch, train_metrics, valid_metrics)

        metric = valid_metrics['balanced_accuracy']

        if metric > best_metric:
            best_metric = metric
            # export_pth(model, OUT_PATH_PTH)
            # export_pt(model, OUT_PATH_PT, CONFIG_SEQUENCE_DIM, CONFIG_CHANNEL_DIM, device=device)
            export_onnx(model, OUT_PATH_ONNX, CONFIG_SEQUENCE_DIM, CONFIG_CHANNEL_DIM, device=device)
            export_mnn(OUT_PATH_ONNX, OUT_PATH_MNN)
            export_onnx(model, OUT_SESSION_PATH_ONNX, CONFIG_SEQUENCE_DIM, CONFIG_CHANNEL_DIM, device=device)
            export_mnn(OUT_SESSION_PATH_ONNX, OUT_SESSION_PATH_MNN)
            print(f'\tbest model metric: {best_metric:.3f}')

    print(best_metric)
    with open(log_metrics_file, 'a') as fout:
        fout.write('Best metric: {:.3}'.format(best_metric))

    return model
