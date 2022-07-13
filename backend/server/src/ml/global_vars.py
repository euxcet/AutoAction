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
