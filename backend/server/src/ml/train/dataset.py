import time
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import ml.data_aug as data_aug
from ml.global_vars import GlobalVars
from ml.filter import Filter


def create_datasets(X:pd.DataFrame, y:pd.Series, test_size=0.25, drop_cols=None, time_dim_first=False):
    ''' Create train and validation datasets, by spliting X into two parts.
    args:
        X: pd.DataFrame, data read from .csv files
        y: pd.Series, labels
        test_size: float, in (0.0, 1.0), the proportion of test data
        drop_cols: drop id columns in X, which are not sensor data
        time_dim_first: wether to transpose the data time dimension
    return:
        tuple: (train dataset, val dataset, label encoder).
        All datasets contain data and labels.
    '''
    # used to transform labels into numbers in [0, num_classes)
    le = LabelEncoder()
    y_enc = le.fit_transform(y) # encoded labels
    # reconstructed data, shape = (samples, length, channels)
    X_grouped:np.ndarray = create_grouped_array(X, drop_cols=drop_cols)
        
    # data augmentation
    if GlobalVars.AUGMENT_EN:
        gain = 1
        strategies = ('scale', 'zoom', 'time warp', 'freq mix')
        data_augmented = []
        y_augmented = []
        # enumerate the unique values in y, at the same time maintain the order
        for group_name in sorted(set(y), key=list(y.values).index):
            # get the indexs corresponding to the group_name
            group_idxs = list(y[y==group_name].index)
            data_augmented.extend([X_grouped[group_idxs], data_aug.augment(
                X_grouped[group_idxs], gain=gain, strategies=strategies)])
            ys = y[y==group_name]
            y_augmented.append(pd.concat([ys] * gain * (2**len(strategies))))
        X_grouped = np.row_stack(data_augmented)
        y_augmented:pd.Series = pd.concat(y_augmented)
        y = y_augmented.reset_index(drop=True)
        y_enc = le.fit_transform(y) # update y encoder
    
    # frequency division
    if GlobalVars.FILTER_EN:
        X_grouped = divide_frequency(X_grouped)

    if time_dim_first:
        # shape = (samples, channels, length)
        X_grouped = X_grouped.transpose(0, 2, 1)
    
    # four np.ndarray
    X_train, X_val, y_train, y_val = train_test_split(X_grouped, y_enc, test_size=test_size)
    # four torch.Tensor
    X_train, X_val = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_val)]
    y_train, y_val = [torch.tensor(arr, dtype=torch.long) for arr in (y_train, y_val)]
    
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    return train_ds, val_ds, le


def create_grouped_array(data:pd.DataFrame, group_col='sample_id', drop_cols=None):
    ''' Reconstruct data by grouping data by group_col.
    args:
        data: pd.DataFrame, [sample_id, measure_id, acc_x, ...]
        group_col: use which column to group data
        drop_cols: drop id columns, which are not sensor data
    return:
        np.ndarray, shape = (sample times, time domain length, channels of all sensors)
    '''
    X_grouped = np.row_stack([group.drop(columns=drop_cols).values[None]
        for _, group in data.groupby(group_col)])
    return X_grouped


def divide_frequency(data:np.ndarray):
    ''' Divide data frequencies using low-pass, band-pass and high-pass filters,
        after data grouping and data augmentation.
    args:
        data: np.ndarray, shape = (samples, length, channels)
    return:
        data after frequency division.
    '''
    # TODO: calc frequencies
    filters = [Filter(mode='low-pass', fs=100, tw=GlobalVars.FILTER_TW,
            fc_low=GlobalVars.FILTER_FC_LOW, window_type=GlobalVars.FILTER_WINDOW),
        Filter(mode='band-pass', fs=100, tw=GlobalVars.FILTER_TW, fc_low=GlobalVars.FILTER_FC_LOW,
            fc_high=GlobalVars.FILTER_FC_HIGH, window_type=GlobalVars.FILTER_WINDOW),
        Filter(mode='high-pass', fs=100, tw=GlobalVars.FILTER_TW, fc_high=GlobalVars.FILTER_FC_HIGH,
            window_type=GlobalVars.FILTER_WINDOW)]
    res = []
    channel_dim = data.shape[2]
    for sample in data:
        divided = []
        for i in range(channel_dim):
            values = sample[:, i]
            divided.append(values)
            for filter in filters:
                divided.append(filter.filter(values))
        res.append(np.array(divided)[np.newaxis,:,:])
    return np.row_stack(res).transpose(0, 2, 1)


def create_test_dataset(X, drop_cols=None):
    ''' This function is never used.
    '''
    X_grouped = np.row_stack([
        group.drop(columns=drop_cols).values[None]
        for _, group in X.groupby('series_id')])
    X_grouped = torch.tensor(X_grouped.transpose(0, 2, 1)).float()
    y_fake = torch.tensor([0] * len(X_grouped)).long()
    return TensorDataset(X_grouped, y_fake)


def create_loaders(train_ds, val_ds, batch_size=512, jobs=0):
    ''' Construct data loaders from datasets
    args:
        train_ds: torch.DataSet, training dataset
        val_ds: torch.DataSet, validation dataset
        batch_size: sample batch size in the data loaders
        jobs: number of cpus to be used
    return:
        tuple: (train data loader, validation data loader)
    '''
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=jobs)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, num_workers=jobs)
    return train_dl, val_dl


def create_train_val_loader(X_TRAIN_PATH, Y_TRAIN_PATH):
    ''' Read csv files and construct data loaders.
    args:
        X_TRAIN_PATH: path of X_train.csv file
        Y_TRAIN_PATH: path of Y_train.csv file
    return:
        tuple: (train data loader, validation data loader)
    '''
    # set csv columns
    ID_COLS = ['sample_id', 'measure_id']
    
    x_cols = {
        'sample_id': np.uint32,
        'measure_id': np.uint32,
    }
    motion_sensors = GlobalVars.MOTION_SENSORS
    for sensor in motion_sensors:
        x_cols[f'{sensor}_x'] = np.float32
        x_cols[f'{sensor}_y'] = np.float32
        x_cols[f'{sensor}_z'] = np.float32
        
    y_cols = {
        'sample_id': np.uint32,
        'group_id': np.uint32,
        'group_name': str,
        'record_id': str
    }
    
    # read .csv data from files
    x_train:pd.DataFrame = pd.read_csv(X_TRAIN_PATH, usecols=x_cols.keys(), dtype=x_cols)
    y_train:pd.DataFrame = pd.read_csv(Y_TRAIN_PATH, usecols=y_cols.keys(), dtype=y_cols)
    
    print('Preparing datasets')
    train_ds, val_ds, le = create_datasets(x_train, y_train['group_name'], drop_cols=ID_COLS)

    batch_size = GlobalVars.BATCH_SIZE
    print(f'Creating data loaders with batch size: {batch_size}')
    train_loader, val_loader = create_loaders(train_ds, val_ds, batch_size, jobs=cpu_count())

    return train_loader, val_loader