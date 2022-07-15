import os
import json
import struct
import random
import matplotlib.pyplot as plt
import numpy as np


'''
Record is parsed from the continuous collected data which may includes several action instances.
Use this class to parse record data from files.
'''
class Record:
    def __init__(self, motion_path, timestamp_path, record_id,
            group_id:int=0, group_name:str='', description:str='',
            cutter=None, cut_data:bool=True):
        ''' Init some parameters.
        '''
        self.motion_path = motion_path  # xxx/Motion_xxx.bin
        self.timestamp_path = timestamp_path    # xxx/Timestamp_xxx.bin
        self.record_id = record_id
        self.group_id = group_id
        self.group_name = group_name
        self.description = description
        self.cutter = cutter
        
        # data
        self.data_labels = ('acc', 'mag', 'gyro', 'linear_acc')
        self.data = None
        self.timestamps = None
        self.cut_data = None

        # self.data, self.timestamps
        self.load_from_file(motion_path, timestamp_path)
        if cut_data:
            self.align_data()
            self.cut()


    def cut(self):
        ''' Use self.cutter to cut the data.
        '''
        self.cut_data = self.cutter.cut(self.data, self.timestamps)


    def align_data(self):
        ''' If the data frequency of all sensors do not match,
            downsample them to align with the lowest frequency.
            Also make sure all sensor data have the same length after aligning.
        '''
        
        data, data_labels = self.data, self.data_labels
        data_t = {label: data[label]['t'] for label in data_labels}
        calc_freq = lambda t: (1e9 * (len(t)-1) / (t[-1]-t[0]))
        data_freq = {label: calc_freq(data_t[label]) for label in data_labels}
        
        for label, freq in data_freq.items():
            print(f'{label} frequency: {freq:.3f} Hz')
            
        freqs = list(data_freq.values())
        min_freq, max_freq = np.min(freqs), np.max(freqs)
        thres = 1.1
        if max_freq / min_freq <= thres:
            print('No need for resampling.')
            return
        
        min_freq = 1e30
        min_freq_label = None
        for label, freq in data_freq.items():
            if freq < min_freq:
                min_freq_label, min_freq = label, freq
                
        for label in data:
            if data_freq[label] / min_freq > thres:
                # downsampling
                print(f'Downsample {label} to {min_freq_label}')
                data[label] = self.down_sample(data[label], data[min_freq_label])
            
        # after resampling
        data_t = {label: data[label]['t'] for label in data_labels}
        data_freq = {label: calc_freq(data_t[label]) for label in data_labels}
        for label, freq in data_freq.items():
            print(f'{label} resampled frequency: {freq:.3f} Hz')
            
        # align to the same length
        min_len = np.min([len(data_t[label]) for label in data_labels])
        for label, sensor_data in data.items():
            if len(data_t[label]) <= min_len: continue
            sensor_data['x'] = sensor_data['x'][:min_len]
            sensor_data['y'] = sensor_data['y'][:min_len]
            sensor_data['z'] = sensor_data['z'][:min_len]
            sensor_data['t'] = sensor_data['t'][:min_len]
            data[label] = sensor_data
        
        self.data = data
        
    
    def down_sample(self, src:dict, ref:dict) -> dict:
        ''' Downsampling src so that it will have the same frequency as refer.
        args:
            src: dict, like {'x': [...], 'y': [...], 'z': [...], 't': [...]},
                all lists are 1D np.ndarray
            ref: the same as arc, with lower frequency
        return:
            A dict, downsampled src.
        '''
        idxs = []
        src_t, ref_t = src['t'], ref['t']
        src_len, ref_len = len(src_t), len(ref_t)
        # preprocess: ensure srt_t[0] < ref_t[idx_start] < ref_t[idx_end] < src_t[-1]
        idx_start, idx_end = 0, ref_len - 1
        while ref_t[idx_start] <= src_t[0] and idx_start < ref_len - 1:
            idx_start += 1
        while ref_t[idx_end] >= src_t[-1] and idx_end > 0:
            idx_end -= 1
        src_idx, ref_idx = 0, idx_start
        while ref_idx <= idx_end:
            t = ref_t[ref_idx]
            while src_t[src_idx] < t:
                src_idx += 1
            # determine which idx is closer
            if src_t[src_idx] - t < t - src_t[src_idx-1]:
                idxs.append(src_idx)
            else: idxs.append(src_idx - 1)
            ref_idx += 1
        idxs = np.array(idxs)
        return {'x': src['x'][idxs], 'y': src['y'][idxs],
                'z': src['z'][idxs], 't': src['t'][idxs]}


    def load_from_file(self, motion_path:str, timestamp_path:str):
        ''' Parse motion data from motion_path file, and store in self.data.
            Parse timestamps from timestamp_path file, and store in self.timestamps
        args:
            motion_path: str, like 'xxx/Motion_xxx.bin'.
            timestamp_path: str, like 'xxx/Timestamp_xxx.json'
        attrs:
            self.data: like {'acc': {'x':[...], 'y':[...], 'z':[...], 't':[...]},
                'mag': {'x':[...], 'y':[...], 'z':[...], 't':[...]},
                'gyro': {'x':[...], 'y':[...], 'z':[...], 't':[...]},
                'linear_acc': {'x':[...], 'y':[...], 'z':[...], 't':[...]},}
            self.timestamps: like [int, int, ...]
        '''
        assert(motion_path.endswith('.bin'))
        assert(timestamp_path.endswith('.json'))
        print(f'Load motion data from file: {motion_path}')
        print(f'Load timestamps from file: {timestamp_path}')
        
        data = {}
        with open(motion_path, 'rb') as f:
            for data_label in ('acc', 'mag', 'gyro', 'linear_acc'):
                size, = struct.unpack('>i', f.read(4))
                xs, ys, zs, ts = [], [], [], []
                for _ in range(size):
                    x, y, z, t = struct.unpack('>fffq', f.read(20))
                    xs.append(x); ys.append(y); zs.append(z); ts.append(t)
                data[data_label] = {'x': np.array(xs, dtype=float), 'y': np.array(ys, dtype=float),
                                    'z': np.array(zs, dtype=float), 't': np.array(ts, dtype=int)}
        self.data = data
        self.timestamps = json.load(open(timestamp_path, 'r'))
                
        print(f'Accelerometer (Number of samples): {len(data["acc"]["t"])}')
        print(f'Magnetic field (Number of samples): {len(data["mag"]["t"])}')
        print(f'Gyroscope (Number of samples): {len(data["gyro"]["t"])}')
        print(f'Linear (Number of samples): {len(data["linear_acc"]["t"])}')
        
    