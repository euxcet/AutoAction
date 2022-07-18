class GlobalVars:
    ''' This variable is used for recording some global properties.
        Note: these variables should only be used in machine learning related
            codes, but not frontend and backend communication.
            For example, motion data sent from frontend contains acc, mag,
            gyro, linear_acc four types, but only part of them will be used to
            train models.
    '''
    # sensor types used in training
    MOTION_SENSORS = ('acc', 'linear_acc', 'gyro')
    # cut window length in time domain
    WINDOW_LENGTH = 200
    # training device: in {'cuda', 'mps', None (cpu)}
    # first check if the device is available, if not, use cpu
    # Note: 'mps' is the apple sillicon GPU backend, which requires macOS Monterey 12.3+
    #   and pytorch 1.13.0(nightly)+, currently is available but has bugs.
    DEVICE = None
    # training batch size
    BATCH_SIZE = 32
    # filter parameters
    FILTER_EN = True
    FILTER_TW = 1
    FILTER_WINDOW = 'hamming'
    FILTER_FC_PEAK = 0.5 # used for the low-pass filter in peak detection
    FILTER_FC_LOW = 6.0
    FILTER_FC_HIGH = 12.0
