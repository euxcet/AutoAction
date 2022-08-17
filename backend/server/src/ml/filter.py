from re import M
import numpy as np
from matplotlib import pyplot as plt

class Filter:
    ''' Implement low-pass, high-pass and band-pass filters for preprocessing signals.
        The band-pass is a combination of low-pass and high-pass.
    '''
    
    def __init__(self, mode:str, fs:float, tw:float, fc_low:float=None,
            fc_high:float=None, window_type:str='hamming'):
        ''' Init filter parameters and calc the window function.
        args:
            mode: str, the filter type, in {'low-pass', 'high-pass', 'band-pass'}.
            fs: float, the samping frequency (Hz) of the input signals to be filtered.
            tw: float, the the transition band width.
            fc_low: float, if mode == 'low-pass', it means the cut-off frequency of
                the filter, else if mode == 'band-pass', it means the lower cut-off frequency.
            fc_high: float, if mode == 'high-pass', it means the cut-off frequency of
                the filter, else if mode == 'band-pass', it means the higher cut-off frequency.
            window_type: str, the filter window type,
                in {'rec', 'hanning', 'hamming', 'blackman'}.
        '''
        # sanity check
        assert mode in ('low-pass', 'high-pass', 'band-pass'), 'Filter bad init: mode'
        if mode in ('low-pass', 'band-pass'):
            assert fc_low is not None, 'Filter bad init: fc_low'
        if mode in ('high-pass', 'band-pass'):
            assert fc_high is not None, 'Filter bad init: fc_high'
        assert window_type in ('rec', 'hanning', 'hamming', 'blackman'), 'Filter bad init: window_type'

        # init attrs
        self.mode = mode; self.fs = fs; self.tw = tw
        self.fc_low = fc_low; self.fc_high = fc_high
        self.window_type = window_type
        self.h = None
        self._init_h()
        
        
    def _init_h(self):
        mode = self.mode
        window_type = self.window_type
        fs, tw = self.fs, self.tw
        fc_low, fc_high = self.fc_low, self.fc_high
        
        # init coef in window function w
        w, n = None, None
        if window_type == 'rec':
            N = ((int(0.91*fs/tw)+1)//2)*2+1
            n = np.arange(N) - (N // 2)
            w = np.ones(N)
        elif window_type == 'hanning':
            N = ((int(3.32*fs/tw)+1)//2)*2+1
            n = np.arange(N) - (N // 2)
            w = 0.5 + 0.5 * np.cos((2*np.pi/(N-1))*n)
        elif window_type == 'hamming':
            N = ((int(3.44*fs/tw)+1)//2)*2+1
            n = np.arange(N) - (N // 2)
            w = 0.54 + 0.46 * np.cos((2*np.pi/(N-1))*n)
        else: # blackman
            N = ((int(5.98*fs/tw)+1)//2)*2+1
            n = np.arange(N) - (N // 2)
            w = 0.42 + 0.5 * np.cos((2*np.pi/(N-1))*n) \
                + 0.08 * np.cos((4*np.pi/(N-1))*n)
           
        # init the unit impulse response h
        h = None  
        if mode == 'low-pass':
            wc = 2 * np.pi * fc_low / fs
            h = np.sin(wc * n) / (np.pi * n)
            h[N // 2] = wc / np.pi
            h *= w
        elif mode == 'high-pass':
            wc = np.pi - 2 * np.pi * fc_high / fs
            h = np.sin(wc * n) / (np.pi * n)
            h[N // 2] = wc / np.pi
            h *= w * np.cos(np.pi * n)
        else: # 'band-pass'
            wc = np.pi * (fc_high - fc_low) / fs
            h = np.sin(wc * n) / (np.pi * n)
            h[N // 2] = wc / np.pi
            h *= 2 * w * np.cos((np.pi * (fc_low+fc_high) / fs) * n)
            
        self.h = h   
    
    
    def filter(self, input:np.ndarray):
        ''' Filter input signals based on the filter parameters initialized before.
        args:
            input: np.ndarray, 1D signals
        return:
            np.ndaray, the filtered signals, with the same length as input.
        '''
        res = np.convolve(input, self.h, mode='same')
        len_res = len(res)
        len_input = len(input)
        if len_res > len_input:
            idx = (len_res - len_input) // 2
            res = res[idx:idx+len_input]
        return res
    
    
if __name__ == '__main__':
    # unit test for Filter
    T = 2
    fs = 100
    t = np.linspace(0, T, num=T*fs)
    
    # full band width signal
    amps = [48, 32, 24, 16, 12, 10, 8, 6, 4, 3, 2, 1]
    freqs = [1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48]
    f = np.zeros(T*fs)
    for amp, freq in zip(amps, freqs):
        f += amp * (np.sin(2*np.pi*freq*t))
    fft_f = np.abs(np.fft.fft(f))
    
    # low-pass (~6Hz) ground truth
    low_amps = [48, 32, 24, 16, 12]
    low_freqs = [1, 2, 3, 4, 6]
    low_f = np.zeros(T*fs)
    for amp, freq in zip(low_amps, low_freqs):
        low_f += amp * (np.sin(2*np.pi*freq*t))
    fft_low_f = np.abs(np.fft.fft(low_f))
    
    # high-pass (24Hz~) ground truth
    high_amps = [3, 2, 1]
    high_freqs = [24, 32, 48]
    high_f = np.zeros(T*fs)
    for amp, freq in zip(high_amps, high_freqs):
        high_f += amp * (np.sin(2*np.pi*freq*t))
    fft_high_f = np.abs(np.fft.fft(high_f))
    
    # band-pass (8Hz~16Hz) ground truth
    band_amps = [10, 8, 6, 4]
    band_freqs = [8, 10, 12, 16]
    band_f = np.zeros(T*fs)
    for amp, freq in zip(band_amps, band_freqs):
        band_f += amp * (np.sin(2*np.pi*freq*t))
    fft_band_f = np.abs(np.fft.fft(band_f))
    
    # filter
    tw = 2
    window = 'hamming'
    low_pass = Filter(mode='low-pass', fs=fs, tw=tw, fc_low=7, window_type=window)
    high_pass = Filter(mode='high-pass', fs=fs, tw=tw, fc_high=20, window_type=window)
    band_pass = Filter(mode='band-pass', fs=fs, tw=tw, fc_low=7, fc_high=20, window_type=window)
    
    # plot
    half_len = len(fft_f) // 2
    plt.subplot(4, 2, 1)
    plt.plot(t, f)
    plt.subplot(4, 2, 2)
    plt.plot(fft_f[:half_len])
    
    plt.subplot(4, 2, 3)
    low_f_filter = low_pass.filter(f)
    print(low_f_filter, low_f_filter.shape)
    print()
    fft_low_f_filter = np.abs(np.fft.fft(low_f_filter))
    plt.plot(t, low_f)
    plt.plot(t, low_f_filter)
    plt.subplot(4, 2, 4)
    plt.plot(fft_low_f[:half_len])
    plt.plot(fft_low_f_filter[:half_len])
    
    plt.subplot(4, 2, 5)
    high_f_filter = high_pass.filter(f)
    print(high_f_filter)
    print()
    fft_high_f_filter = np.abs(np.fft.fft(high_f_filter))
    plt.plot(t, high_f)
    plt.plot(t, high_f_filter)
    plt.subplot(4, 2, 6)
    plt.plot(fft_high_f[:half_len])
    plt.plot(fft_high_f_filter[:half_len])
    
    plt.subplot(4, 2, 7)
    band_f_filter = band_pass.filter(f)
    print(band_f_filter)
    print()
    fft_band_f_filter = np.abs(np.fft.fft(band_f_filter))
    plt.plot(t, band_f)
    plt.plot(t, band_f_filter)
    plt.subplot(4, 2, 8)
    plt.plot(fft_band_f[:half_len])
    plt.plot(fft_band_f_filter[:half_len])
    
#plt.show()
            
        
