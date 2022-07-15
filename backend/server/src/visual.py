# this file is used for debugging
import os
import numpy as np
from matplotlib import pyplot as plt

import file_utils
from ml.record import Record
from ml.cutter.peak_cutter import PeakCutter


def plot_data(data:dict, sensors:tuple=('acc',), idx_range:tuple=None,
        title:str=None, timestamps:list=None):
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
        plt.plot(t, x); plt.plot(t, y); plt.plot(t, z); plt.plot(t, norm)
        
        if timestamps:
            xyz = np.concatenate([x, y, z])
            min_val, max_val = np.min(xyz), np.max(norm)
            for t in timestamps:
                plt.plot([t,t], [min_val,max_val], '--', color='black')
    
        plt.ylabel(sensor)
        plt.legend(['x', 'y', 'z', 'norm'], loc='lower right')
        
    if title: plt.suptitle(title)
    plt.show()


if __name__ == '__main__':
    task_list_id = 'TL13r912je' # default task list
    task_id = 'TKh01oe3tq' # Task-3
    subtask_id = 'STkwmtabqe'
    record_id = 'RD39l3hjqv'
    record_path = file_utils.get_record_path(task_list_id, task_id, subtask_id, record_id)
    print(record_path)
    motion_path, timestamp_path = None, None
    for filename in os.listdir(record_path):
        if filename.startswith('Motion'):
            motion_path = os.path.join(record_path, filename)
        if filename.startswith('Timestamp'):
            timestamp_path = os.path.join(record_path, filename)
            
    # timestamps
    
    cutter = PeakCutter('linear_acc', 80, 200, 20)
    record = Record(motion_path, timestamp_path, record_id,
        group_id=0, group_name='Task-0', cutter=cutter, cut_data=True)
    data = record.data

    timestamps = record.timestamps
    plot_data(data, sensors=('acc', 'linear_acc', 'gyro'), timestamps=timestamps)