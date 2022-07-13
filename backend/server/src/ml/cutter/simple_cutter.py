import numpy as np
from ml.cutter.range_cutter import RangeCutter

class SimpleCutter():
    ''' Cut data simply by timestamp boundaries.
    '''
    def __init__(self):
        ''' Nothing to init.
        '''
        pass

    def cut(self, data, timestamp):
        ''' Cut the sensors data by timestamp boundaries.
        args:
            data: like {'acc': {'x':[...], 'y':[...], 'z':[...], 't':[...]},
                'mag': {'x':[...], 'y':[...], 'z':[...], 't':[...]},
                'gyro': {'x':[...], 'y':[...], 'z':[...], 't':[...]},
                'linear_acc': {'x':[...], 'y':[...], 'z':[...], 't':[...]},},
                all lists for each sensor have the same length
            timestamp: the timestamp series at the beginning of n action sampling,
                like [t_0, t_1, ..., t_{n-1}]
        return:
            Sensors data after cutting.
        '''
        range_cutter = RangeCutter()
        cut_data = {}
        for label in data:
            cut_range = self.cut_range(data[label], timestamp)
            cut_data[label] = range_cutter.cut(data[label], cut_range)
        return cut_data
    

    def cut_range(self, data, timestamp):
        ''' Use the timestamps to determine the cut ranges of all sensor data.
            In this simple strategy, simply calc cut ranges for each sensor
            individually only by timestamp boundaries.
        args:
            data: a type of sensor data, like {'x':[...], 'y':[...], 'z':[...], 't':[...]}
            timestamp: the same as self.cut()
        return:
            Cut ranges for each sensor, like [(start:int, end:int), ...], [start, end)
        '''
        bound = timestamp + [timestamp[-1] + 1e20]
        data_t = data['t']
        offset = 0
        while data_t[offset] < bound[0]:
            offset += 1
            
        res = []
        bound_idx = 1
        start, end = 0, None
        for i, t in enumerate(data_t[offset:]):
            if t < bound[bound_idx]: continue
            end = i
            res.append((start+offset, end+offset))
            start = i
            bound_idx += 1
        res.append((start+offset, len(data_t)))
        return res
    

    def to_json(self):
        ''' Return the info in json format.
        '''
        return {'name': 'SimpleCutter', 'param': []}
        

if __name__ == '__main__':
    # test cut_range
    data = {'t': np.arange(100)}
    timestamp = [10, 25, 40, 60, 90]
    cutter = SimpleCutter()
    cut_range = cutter.cut_range(data, timestamp)
    for pair in cut_range:
        print(pair)
