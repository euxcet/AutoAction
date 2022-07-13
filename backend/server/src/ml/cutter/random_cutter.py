import numpy as np
from ml.cutter.range_cutter import RangeCutter
from ml.cutter.simple_cutter import SimpleCutter

class RandomCutter():
    def __init__(self, length=128):
        ''' Init the cutter.
        '''
        self.length = length

    def cut(self, data:dict, timestamp:list):
        ''' Cut the sensors data by random ranges.
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
        length = self.length
        range_cutter = RangeCutter()
        simple_cutter = SimpleCutter()
        cut_data = {}
        anchor = list(data.keys())[0]
        
        cut_range = []
        simple_cut_range = simple_cutter.cut_range(data[anchor], timestamp)
        for start, end in simple_cut_range:
            r = np.random.randint(start, end - length)
            cut_range.append(r, r + length)
        for label in data:
            cut_data[label] = range_cutter.cut(data[label], cut_range)
        return cut_data
    

    def to_json(self):
        ''' Return the info in json format.
        '''
        return {'name': 'RandomCutter', 
            'param': [{'name': 'length', 
                    'type': 'int', 
                    'description': 'Length of randomly cut samples', 
                    'default': 128
                }]
            }
