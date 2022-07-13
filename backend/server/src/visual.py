# this file is used for debugging
import numpy as np
from matplotlib import pyplot as plt

from ml.record import Record
from ml.cutter.peak_cutter import PeakCutter


def plot_data(data, sensors=('acc',), idx_range=None, title=None):
    for i, sensor in enumerate(sensors):
        plt.subplot(len(sensors), 1, i+1)
        sensor_data = data[sensor]
        if idx_range is None:
            idx_range = (0, 1)
        data_len = len(sensor_data['t'])
        start = int(idx_range[0] * data_len)
        end = int(idx_range[1] * data_len)
        x = sensor_data['x'][start:end]
        y = sensor_data['y'][start:end]
        z = sensor_data['z'][start:end]
        t = sensor_data['t'][start:end]
        norm = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        plt.plot(t, x)
        plt.plot(t, y)
        plt.plot(t, z)
        plt.ylabel(sensor)
        plt.plot(t, norm)
        plt.legend(['x', 'y', 'z', 'norm'], loc='lower right')
        
    if title: plt.suptitle(title)
    plt.show()


if __name__ == '__main__':
    filename = '../data/record/TL13r912je/TKu0l0pg2n/STyww0isly/RD9d2p417j/Motion_1657461221050.bin'
    timestamp_filename = '../data/record/TL13r912je/TKu0l0pg2n/STyww0isly/RD9d2p417j/Timestamp_1657461221050.json'
    cutter = PeakCutter('acc', 100, 300, 10)
    record = Record(filename, timestamp_filename, record_id='', cutter=cutter)
    data = record.data
    plot_data(data, sensors=('acc', 'linear_acc', 'gyro'), idx_range=(0.5, 0.55), title='Shaking')