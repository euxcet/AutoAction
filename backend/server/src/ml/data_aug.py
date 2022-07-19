import time
import numpy as np
from matplotlib import pyplot as plt

from ml.filter import Filter
from ml.global_vars import GlobalVars


filters = []


def init_filters():
    global filters
    if not filters:
        print('init')
        filters = [Filter(mode='low-pass', fs=100, tw=GlobalVars.FILTER_TW,
            fc_low=GlobalVars.FILTER_FC_LOW, window_type=GlobalVars.FILTER_WINDOW),
        Filter(mode='band-pass', fs=100, tw=GlobalVars.FILTER_TW, fc_low=GlobalVars.FILTER_FC_LOW,
            fc_high=GlobalVars.FILTER_FC_HIGH, window_type=GlobalVars.FILTER_WINDOW),
        Filter(mode='high-pass', fs=100, tw=GlobalVars.FILTER_TW, fc_high=GlobalVars.FILTER_FC_HIGH,
            window_type=GlobalVars.FILTER_WINDOW)]


def scale(data:np.ndarray, gain:int=1, s_mean:float=1.0, s_std:float=0.2,
    s_min:float=0.5, s_max:float=2.0):
    ''' Randomly scale data for each sample.
        Scaling factor s ~ N(s_mean, s_std), in range [s_min, s_max]
    args:
        data: np.ndarray, shape = (samples, length, channels).
            All samples must be from the same group.
        gain: int, the augmented data ratio want to generate compared with
            the original data, must be int for easier implementation.
        s_mean: float, mean of s.
        s_std: float, std of s.
        s_min: float, the min value of s.
        s_max: float, the max value of s.
    return:
        Augmented data, with samples = gain * len(data).
    '''
    res = []
    samples = data.shape[0]
    for _ in range(int(gain)):
        s = np.random.randn(samples) * s_std + s_mean
        # make sure s is within [s_min, s_max]
        while np.min(s) < s_min or np.max(s) > s_max:
            mask = (s < s_min) | (s > s_max)
            s[mask] = np.random.randn(np.sum(mask)) * s_std + s_mean
        res.append(data * s[:, np.newaxis, np.newaxis])
    return np.row_stack(res)


def zoom(data:np.ndarray, gain:int=1, r1:float=0.8, r2:float=1.2):
    ''' Zoom the data with respect to center position in time domain.
        Zooming factor z ~ U(r[0], r[1]), z > 1.0 means faster action.
        If z > 1.0, using 0.0 as padding.
        Using linear interpolation to generate data values at unknown time.
    args:
        data: np.ndarray, shape = (samples, length, channels).
            All samples must be from the same group.
        gain: int, the augmented data ratio want to generate compared with
            the original data, must be int for easier implementation.
        r1, r2: float, the lower and upper bound of zooming factor distribution.
    return:
        Augmented data, with samples = gain * len(data).
    '''
    samples, length, channels = data.shape
    half = length / 2
    xp = np.arange(length)
    res = []
    for _ in range(int(gain)):
        zs = np.random.uniform(r1, r2, samples)
        for sample, z in zip(data, zs): # sample.shape = (length, channels)
            x = np.linspace((1-z)*half, (1+z)*half, num=length)
            zoomed = np.row_stack([np.interp(x, xp, sample[:,i],
                left=0.0, right=0.0) for i in range(channels)])
            res.append(zoomed[np.newaxis,:,:])
    np.row_stack(res).transpose(0, 2, 1)
    return np.row_stack(res).transpose(0, 2, 1)


def freq_mix(data:np.ndarray, gain:int=1):
    ''' First separate data frequencies using low-pass, band-pass
        and high-pass filters, then shuffle differt band width data
        and reassemble them to produce augmented data.
    args:
        data: np.ndarray, shape = (samples, length, channels).
            All samples must be from the same group.
        gain: int, the augmented data ratio want to generate compared with
            the original data, must be int for easier implementation.
    return:
        Augmented data, with samples = gain * len(data).
    '''
    init_filters(); global filters
    channels = data.shape[2]
    data_filtered = []
    for filter in filters:
        filtered = []
        for sample in data:
            filtered.append(np.row_stack([filter.filter(sample[:,i])
                for i in range(channels)])[np.newaxis,:,:])
        data_filtered.append(np.row_stack(filtered).transpose(0,2,1)[np.newaxis,:,:,:])
    data_filtered = np.row_stack(data_filtered)
    
    res = []
    for _ in range(int(gain)):
        # only shuffle data filtered by band-pass and high-pass filters
        for idx in range(1, len(filters)):
            np.random.shuffle(data_filtered[idx])
        res.append(np.sum(data_filtered, axis=0))
    return np.row_stack(res)


def augment(data:np.ndarray, gain:int=1, strategies:tuple=('scale', 'zoom', 'freq mix')):
    ''' Augment data using the combinations of strategies.
        TODO: implement 'time warp' strategy.
    args:
        data: np.ndarray, shape = (samples, length, channels).
            All samples must be from the same group.
        gain: int, the gain used in the first strategy in each combination,
            gains in other stategies always equals to 1.
        strategies: tuple of str, the strategies to be used in combinations,
            each strategy must in {'scale', 'zoom', 'freq mix'}.
    return:
        The augmented data, with length = gain * len(data) * (2 ^ len(strategies) - 1).
    '''
    # sanity check
    for strategy in strategies:
        assert strategy in {'scale', 'zoom', 'freq mix'}, f'Bad augment strategy: {strategy}'
    res = []
    len_strategies = len(strategies)
    for code in range(1, 2**len_strategies): # encode combinations
        first = True # gain = gain in the first strategy, 1 in others
        current_data = data.copy()
        for i, strategy in enumerate(strategies):
            if (code & (2**i)) == 0: continue
            current_gain = 1
            if first: current_gain = gain; first = False
            if strategy == 'scale':
                current_data = scale(data, gain=current_gain)
            elif strategy == 'zoom':
                current_data = zoom(data, gain=current_gain)
            else: current_data = freq_mix(data, gain=current_gain)
        res.append(current_data)
    return np.row_stack(res)

        
if __name__ == '__main__':
    data = np.random.randn(100, 300, 9)
    tic = time.perf_counter()
    # res = scale(data, gain=1)
    # res = zoom(data, gain=1)
    # res = freq_mix(data, gain=1)
    res = augment(data, gain=1)
    toc = time.perf_counter()
    print(f'### time: {(toc-tic)*1000:.1f} ms')
    
    print(res.shape)