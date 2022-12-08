import time
import numpy as np
import random
from scipy import interpolate as interp
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

from ml.filter import Filter
from ml.global_vars import GlobalVars


filters = []


def init_filters():
    global filters
    if not filters:
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


def zoom(data:np.ndarray, gain:int=1, r1:float=0.9, r2:float=1.0):
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
            x = np.linspace((1-z)*half, (1+z)*half, num=length, endpoint=False)
            zoomed = np.row_stack([np.interp(x, xp, sample[:,i],
                left=0.0, right=0.0) for i in range(channels)])
            res.append(zoomed[np.newaxis,:,:])
    return np.row_stack(res).transpose(0, 2, 1)


def time_warp(data:np.ndarray, gain:int=1, k_std:float=0.05):
    ''' Use time warping strategy to augment the data.
    args:
        data: np.ndarray, shape = (samples, length, channels).
            All samples must be from the same group.
        gain: int, the augmented data ratio want to generate compared with
            the original data, must be int for easier implementation.
        k_std: float, knot std, defines the std of knots, which are
            used to generate random spline curve for time warping.
    '''
    samples, length, channels = data.shape
    kx = np.array([0, 1/3, 2/3, 1])
    kx_new = np.linspace(0, 1, num=length, endpoint=False)
    x = np.arange(length)
    res = []
    for _ in range(int(gain)):
        kys = np.ones((samples, 4))
        kys[:, 1:3] += np.random.randn(samples, 2) * k_std
        for sample, ky in zip(data, kys):
            # cubic spline interpolation
            tck = interp.splrep(kx, ky, s=0, per=True)
            ky_new = interp.splev(kx_new, tck, der=0) * kx_new * length
            # linear interpolation
            warped = np.row_stack([np.interp(x=ky_new, xp=x,
                fp=sample[:,i]) for i in range(channels)])
            res.append(warped[np.newaxis,:,:])
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


def augment(data:np.ndarray, gain:int=1, strategies:tuple=('scale', 'zoom', 'time warp')):
    ''' Augment data using the combinations of strategies.
    args:
        data: np.ndarray, shape = (samples, length, channels).
            All samples must be from the same group.
        gain: int, the gain used in the first strategy in each combination,
            gains in other stategies always equals to 1.
        strategies: tuple of str, the strategies to be used in combinations,
            each strategy must in {'scale', 'zoom', 'time warp', 'freq mix'}.
    return:
        The augmented data, with length = gain * len(data) * (2 ^ len(strategies) - 1).
    '''
    # sanity check
    for strategy in strategies:
        assert strategy in {'scale', 'zoom', 'time warp', 'freq mix'}, \
            f'Bad augment strategy: {strategy}'
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
                current_data = scale(current_data, gain=current_gain)
            elif strategy == 'zoom':
                current_data = zoom(current_data, gain=current_gain)
            elif strategy == 'time warp':
                current_data = time_warp(current_data, gain=current_gain)
            else: current_data = freq_mix(current_data, gain=current_gain)
        res.append(current_data)
    return np.row_stack(res)

def rotate(data:np.ndarray):
    ''' Augment data by rotation.
    args:
        data: np.ndarray, shape = (samples, length, channels).
            All samples must be from the same group.
    return:
        The augmented data, with the same length as input

    '''
    mat = R.from_rotvec(np.array([random.uniform(-20, 20), random.uniform(-20, 20), random.uniform(-20, 20)]), degrees=True)
    mat = R.random()
    p = data.reshape(-1, 3)
    pr = mat.apply(p)
    pr = pr.reshape(data.shape)
    return pr

        
if __name__ == '__main__':
    data = np.random.randn(1000, 200, 9)
    tic = time.perf_counter()
    # res = scale(data, gain=1)
    # res = zoom(data, gain=1)
    # res = time_warp(data, gain=1)
    # res = freq_mix(data, gain=1)
    res = augment(data, gain=1)
    toc = time.perf_counter()
    print(f'### time: {(toc-tic)*1000:.1f} ms')
    print(res.shape)