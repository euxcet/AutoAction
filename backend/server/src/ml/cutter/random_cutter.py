from ml.cutter.range_cutter import RangeCutter
import random

class RandomCutter():
    def __init__(self, length=128):
        self.length = length
        self.range_cutter = RangeCutter()

    def cut(self, sensors_data, timestamp):
        cut_range = self.cut_(sensors_data[0], timestamp)
        return self.range_cutter.cut(sensors_data, cut_range)

    def cut_(self, sensor_data, timestamp):
        t_index = 0
        pre = 0
        cut_range = []
        for i in range(len(sensor_data)):
            if sensor_data[i][3] < timestamp[t_index]:
                continue
            if sensor_data[i][3] >= timestamp[t_index + 1] or i == len(sensor_data) - 1:
                cut_range.append((pre, i - 1))
                pre = i
                t_index += 1
        final_cut_range = []
        for r in cut_range:
            l = r[1] - r[0] - self.length
            start_pos = random.randint(0, l)
            final_cut_range.append((start_pos, start_pos + self.length))

        return final_cut_range

    def to_json(self):
        return {'name': 'RandomCutter', 'param': [{'name': 'length', 'type': 'int', 'description': 'Length of randomly cut samples', 'default': 128}]}