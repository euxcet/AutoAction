import numpy as np
from matplotlib import pyplot as plt

class RangeCutter():
    def __init__(self):
        ''' Nothing to init.
        '''
        pass

    def cut(self, data, cut_range):
        ''' Cut all data in sensors_data with the range in cut_range.
        args:
            data: like {'x':[...], 'y':[...], 'z':[...], 't':[...]}
            cut_range: like [(start:int, end:int), ...]
        return:
            Sensors data after cutting.
        '''
        res = {}
        for label, vec in data.items():
            v = []
            for r in cut_range:
                v.append(vec[r[0]:r[1]])
            res[label] = np.array(v)
        return res