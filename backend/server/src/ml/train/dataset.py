import torch
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

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


def create_loaders(train_ds, valid_ds, batch_size=512, jobs=0):
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, batch_size, shuffle=False, num_workers=jobs)
    return train_dl, valid_dl


def create_train_val_loader(TRAIN_X_PATH, TRAIN_Y_PATH):
    ID_COLS = ['series_id', 'measurement_number']

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
        'group_name': str,
        'record_id': str
    }
    x_train = pd.read_csv(TRAIN_X_PATH, usecols=x_cols.keys(), dtype=x_cols)
    y_train = pd.read_csv(TRAIN_Y_PATH, usecols=y_cols.keys(), dtype=y_cols)

    print('Preparing datasets')
    train_dataset, val_dataset, enc = create_datasets(x_train, y_train['group_name'], drop_cols=ID_COLS)

    bs = 8
    print(f'Creating data loaders with batch size: {bs}')
    train_loader, val_loader = create_loaders(train_dataset, val_dataset, bs, jobs=cpu_count())

    return train_loader, val_loader