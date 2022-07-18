from ast import Global
import numpy as np
from matplotlib import pyplot as plt
from ml.cutter.range_cutter import RangeCutter
from ml.filter import Filter
from ml.global_vars import GlobalVars

class PeakCutter():
    ''' Cut by amplitude peaks.
    '''
    def __init__(self, anchor:str, forward=100, length=200, noise=20):
        ''' Init the cutter with some basic parameters.
        '''
        self.anchor = anchor # anchor = 'acc'
        self.forward = forward
        self.length = length
        self.noise = noise

    def cut(self, data, timestamp):
        ''' Cut the sensors data by amplitude peaks.
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
        anchor = self.anchor
        assert(anchor in data)
        range_cutter = RangeCutter()
        cut_range = self.cut_range(data[anchor], timestamp)
        cut_data = {}
        for label in data:
            cut_data[label] = range_cutter.cut(data[label], cut_range)
        return cut_data
    
    
    def cut_range(self, data, timestamp):
        ''' Use the timestamps and amplitude peaks to determine the cut ranges. 
        args:
            data: a type of sensor data, like {'x':[...], 'y':[...], 'z':[...], 't':[...]}
            timestamp: the same as self.cut()
        return:
            Cut ranges for each sensor, like [(start:int, end:int), ...], [start, end)
        '''
        forward = self.forward
        length = self.length
        noise = self.noise
        randint = np.random.randint
        bound = timestamp + [timestamp[-1] + 1e20]
        data_t = data['t']
        freq = 1e9 * (len(data_t)-1) / (data_t[-1] - data_t[0])
        value_keys = list(data.keys())
        value_keys.remove('t')  # ['x', 'y', 'z']
        norm = np.sum(np.vstack([np.square(data[key]) for key in value_keys]), axis=0)
        # low-pass filter
        low_pass = Filter(mode='low-pass', fs=freq, tw=GlobalVars.FILTER_TW,
            fc_low=GlobalVars.FILTER_FC_PEAK, window_type=GlobalVars.FILTER_WINDOW)
        norm_filtered = low_pass.filter(norm)
        
        offset = 0
        while data_t[offset] < bound[0]:
            offset += 1
            
        res = []
        bound_idx = 1
        start, end = 0, None
        for i, t in enumerate(data_t[offset:]):
            if t < bound[bound_idx]: continue
            end = i
            peak_idx = start + np.argmax(norm_filtered[start+offset:end+offset])
            # note: first min then max, the order matters!
            peak_start = min(end - length, peak_idx - forward + randint(-noise, noise+1))
            peak_start = max(0, peak_start)
            res.append((offset+peak_start, offset+peak_start+length))
            start = i
            bound_idx += 1
        end = len(data_t) - offset
        peak_idx = start + np.argmax(norm_filtered[start+offset:end+offset])
        peak_start = min(end - length, peak_idx - forward + randint(-noise, noise+1))
        peak_start = max(0, peak_start)
        res.append((offset+peak_start, offset+peak_start+length))
        return res


    def to_json(self):
        ''' Return the required info in json format.
        '''
        return {'name': 'PeakCutter',
            'param': [{'name': 'anchor',
                    'type': 'str',
                    'description': 'Use what data as the amplitude reference.',
                    'default': 'acc'
                }, {'name': 'forward',
                    'type': 'int',
                    'description': 'Position of the peak in the sample',
                    'default': 80
                }, {'name': 'length', 
                    'type': 'int', 
                    'description': 'Length of cut samples', 
                    'default': 128
                }, {'name': 'noise',
                    'type': 'int', 
                    'description': 'Random disturbance size of peak position', 
                    'default': 10
                }]
            }
        
        
if __name__ == '__main__':
    # test for the peak cutter
    y_sin = np.sin(np.linspace(0, np.pi, 40))
    signal = []
    for _ in range(8):
        start = np.random.randint(0, 260)
        base = np.zeros(300)
        base[start:start+40] += y_sin
        signal.append(base)
    signal = np.concatenate(signal)
    data = {
        'acc': {
            'x': signal.copy(),
            'y': signal.copy(),
            'z': signal.copy(),
            't': np.arange(8 * 300),
        }
    }
    timestamp = [0, 300, 600, 900, 1200, 1500, 1800, 2100]
    peak_cutter = PeakCutter(anchor='acc', forward=50, length=100, noise=5)
    cut_data = peak_cutter.cut(data, timestamp)
    
    for clip in cut_data['acc']['x']:
        plt.plot(clip)
    plt.show()
