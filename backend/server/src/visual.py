# this file is used for debugging
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

import file_utils
from ml.record import Record
from ml.cutter.peak_cutter import PeakCutter
from ml.train.dataset import create_train_val_loader
from ml.filter import Filter

import pandas as pd

def plot_data(data:dict, sensors:tuple=('acc',), idx_range:tuple=None,
        title:str=None, timestamps:list=None):
    for i, sensor in enumerate(sensors):
        # plot the signals in time domain
        plt.subplot(len(sensors), 2, i+i+1)
        sensor_data = data[sensor]
        if idx_range is None:
            idx_range = (0, 1)
        data_len = len(sensor_data['t'])
        start = int(idx_range[0] * data_len)
        end = int(idx_range[1] * data_len)
        x = sensor_data['x'][start:end]
        y = sensor_data['y'][start:end]
        z = sensor_data['z'][start:end]
        ts = sensor_data['t'][start:end]
        t = [i for i in range(len(ts))]
        norm = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        plt.plot(t, x); plt.plot(t, y); plt.plot(t, z); plt.plot(t, norm)
        
        if timestamps:
            xyz = np.concatenate([x, y, z])
            min_val, max_val = np.min(xyz), np.max(norm)
            for t in timestamps:
                pos = np.searchsorted(ts, t)
                plt.plot([pos,pos], [min_val,max_val], '--', color='black')

        plt.ylabel(sensor)
        plt.legend(['x', 'y', 'z', 'norm'], loc='lower right')
        
        # plot the signals in frequency domain
        plt.subplot(len(sensors), 2, i+i+2)
        fft_x = np.fft.fft(x)
        fft_y = np.fft.fft(y)
        fft_z = np.fft.fft(z)
        len_half = len(fft_x) // 2
        plt.plot(np.abs(fft_x[:len_half])); plt.plot(np.abs(fft_y[:len_half]))
        plt.plot(np.abs(fft_z[:len_half]))
        
    if title: plt.suptitle(title)
    plt.show()
    
    
def plot_filter(data:dict, sensor:str, idx_range:tuple=None, timestamps:list=None):
    sensor_data = data[sensor]
    if idx_range is None:
        idx_range = (0.0, 1.0)
    data_len = len(sensor_data['t'])
    start = int(idx_range[0] * data_len)
    end = int(idx_range[1] * data_len)
    x = sensor_data['x'][start:end]
    y = sensor_data['y'][start:end]
    z = sensor_data['z'][start:end]
    t = sensor_data['t'][start:end]
    norm = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    
    ylim = 80
    plt.subplot(4, 1, 1)
    plt.plot(t, x); plt.plot(t, y); plt.plot(t, z); plt.plot(t, norm)
    plt.ylim(-ylim, ylim)
    
    f1, f2 = 0.5, 16
    tw = 1
    window = 'hamming'
    filters = [Filter(mode='low-pass', fs=100, tw=tw, fc_low=f1, window_type=window),
               Filter(mode='band-pass', fs=100, tw=tw, fc_low=f1, fc_high=f2, window_type=window),
               Filter(mode='high-pass', fs=100, tw=tw, fc_high=f2, window_type=window)]
    for i, filter in enumerate(filters):
        plt.subplot(4, 1, i+2)
        plt.plot(t, filter.filter(x)); plt.plot(t, filter.filter(y))
        plt.plot(t, filter.filter(z)); plt.plot(t, filter.filter(norm))
        plt.ylim(-ylim, ylim)
        
    plt.show()

def visual_record(task_id, subtask_id, record_id):
    task_list_id = 'TL13r912je' # default task list
    record_path = file_utils.get_record_path(task_list_id, task_id, subtask_id, record_id)
    print(record_path)
    motion_path, timestamp_path = None, None
    for filename in os.listdir(record_path):
        if filename.startswith('Motion'):
            motion_path = os.path.join(record_path, filename)
        if filename.startswith('Timestamp'):
            timestamp_path = os.path.join(record_path, filename)
            
    # timestamps
    cutter = PeakCutter('linear_acc', 60, 100, 10)
    record = Record(motion_path, timestamp_path, record_id,
        group_id=0, group_name='Task-0', cutter=cutter, cut_data=True)
    data = record.data
    timestamps = record.timestamps
    print(data)
    
    plot_data(data, sensors=('acc', 'linear_acc', 'gyro'), timestamps=timestamps)
    #plot_filter(data, 'linear_acc', idx_range=(0.0, 1.0))


def visual_dataset(train_id:str, sensors:tuple=('acc', 'linear', 'gyro'), sample_count:int=10, class_ids:list=[0]):
    ROOT = file_utils.get_train_path(train_id)
    X_TEST_PATH = os.path.join(ROOT, 'X_test.csv')    # train_data
    Y_TEST_PATH = os.path.join(ROOT, 'Y_test.csv')    # train_labels

    train_loader, val_loader = create_train_val_loader(X_TEST_PATH, Y_TEST_PATH, 'label.txt')

    data = [[] for i in range(len(class_ids))]

    current_sample_count = np.zeros(shape=max(class_ids)+1, dtype=np.int)

    for x_batch, y_batch in train_loader:
        x = x_batch.numpy()
        
        for i in range(x_batch.shape[0]):
            if y_batch[i] in class_ids and current_sample_count[y_batch[i]] < sample_count:
                current_sample_count[y_batch[i]] += 1
                data[class_ids.index(y_batch[i])].append(x[i, :, 0::4])

    sample_length = data[0][0].shape[0]

    figure_row = len(sensors)
    figure_col = len(class_ids)
    for i in range(len(class_ids)):
        d = np.array(data[i]).transpose([2, 0, 1])
        for j, sensor in enumerate(sensors):
            plt.subplot(figure_row, figure_col, j * figure_col + i + 1)
            if sensor == 'acc':
                xyz = d[0:3]
            elif sensor == 'linear':
                xyz = d[3:6]
            elif sensor == 'gyro':
                xyz = d[6:9]
            xyz = xyz.reshape(xyz.shape[0], -1)
            t = np.arange(0, xyz.shape[1])
            plt.plot(t, xyz[0])
            plt.plot(t, xyz[1])
            plt.plot(t, xyz[2])

            min_val, max_val = np.min(xyz), np.max(xyz)
            for divider in range(0, len(t) + 1, sample_length):
                plt.plot([divider,divider], [min_val,max_val], '--', color='black')


    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default = '../data', help='Root directory of raw data.')
    parser.add_argument('--task_id', type=str, default = 'TK7t0oxh08', help='Task id of the record.')
    parser.add_argument('--subtask_id', type=str, default = 'STh1cx490s', help='Subtask id of the record.')
    parser.add_argument('--record_id', type=str, default = 'RD0', help='Record id of the record.')
    args = parser.parse_args()
    file_utils.set_data_root(args.data_root)

    #visual_record()
    #visual_dataset('XT9me9xq7y', sample_count=10, class_ids=[1, 2])
    visual_record(args.task_id, args.subtask_id, args.record_id)

