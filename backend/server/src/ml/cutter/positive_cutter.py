from ml.filter import Filter
from ml.global_vars import GlobalVars
from ml.cutter.range_cutter import RangeCutter
import torch
import numpy as np
import file_utils

class PositiveCutter():
    def __init__(self, model=None, negative_id=-1):
        self.model = model
        self.filters = [Filter(mode='low-pass', fs=100, tw=GlobalVars.FILTER_TW,
                fc_low=GlobalVars.FILTER_FC_LOW, window_type=GlobalVars.FILTER_WINDOW),
            Filter(mode='band-pass', fs=100, tw=GlobalVars.FILTER_TW, fc_low=GlobalVars.FILTER_FC_LOW,
                fc_high=GlobalVars.FILTER_FC_HIGH, window_type=GlobalVars.FILTER_WINDOW),
            Filter(mode='high-pass', fs=100, tw=GlobalVars.FILTER_TW, fc_high=GlobalVars.FILTER_FC_HIGH,
                window_type=GlobalVars.FILTER_WINDOW)]
        self.negative_id = negative_id

    def filter_data(self, data, s):
        origin = [np.array(data[t][s:s+GlobalVars.WINDOW_LENGTH]) for t in ['x', 'y', 'z']]
        result = []
        for d in origin:
            result.append(d)
            for f in self.filters:
                result.append(f.filter(d))
        return np.array(result)

    def cut(self, data, timestamp):
        cut_range = [(0, GlobalVars.WINDOW_LENGTH)]
        self.model.eval()
        length = data['acc']['x'].shape[0]
        last_positive = -100
        for s in range(0, length - GlobalVars.WINDOW_LENGTH):
            acc_data = self.filter_data(data['acc'], s)
            linear_data = self.filter_data(data['linear_acc'], s)
            gyro_data = self.filter_data(data['gyro'], s)
            input_data = np.concatenate((acc_data, linear_data, gyro_data), axis=0).T
            input_data = input_data[np.newaxis, :].astype(np.float32)
            input_data = torch.from_numpy(input_data).to(GlobalVars.DEVICE)
            answer = self.model(input_data)
            # TODO read labels from label.txt
            if torch.argmax(answer) != 0 and s > last_positive + 10:
                last_positive = s
                cut_range.append((s, s + GlobalVars.WINDOW_LENGTH))

        print(len(cut_range))
        range_cutter = RangeCutter()
        cut_data = {}
        for label in data:
            cut_data[label] = range_cutter.cut(data[label], cut_range)
        return cut_data