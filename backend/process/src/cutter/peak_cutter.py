from cutter.range_cutter import RangeCutter
import random

class PeakCutter():
    def __init__(self, anchor, forward_extension = 80, length = 128, noise = 10):
        self.anchor = anchor
        self.range_cutter = RangeCutter()
        self.forward_extension = forward_extension
        self.length = length
        self.noise = noise

    # sensor_data: [acc, gyro]
    def cut(self, sensors_data, timestamp):
        cut_range = self.cut_by_peak(sensors_data[self.anchor], timestamp)
        return self.range_cutter.cut(sensors_data, cut_range)


    def cut_by_peak(self, sensor_data, timestamp):
        result = []
        current = []
        t_index = 0
        pre = 0
        cut_range = []
        for i in range(len(sensor_data)):
            if sensor_data[i][3] < timestamp[t_index]:
                continue
            if sensor_data[i][3] >= timestamp[t_index + 1] or i == len(sensor_data) - 1:
                cut_range.append((pre, i - 1))
                result.append(current)
                pre = i
                current = []
                t_index += 1
            if sensor_data[i][3] >= timestamp[t_index] and sensor_data[i][3] < timestamp[t_index + 1]:
                current.append(sensor_data[i])

        final_cut_range = []
        for clip, cr in zip(result, cut_range):
            peak_pos = 0
            peak_max = 0
            for i in range(len(clip)):
                v = clip[i]
                amplitude = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
                if amplitude > peak_max:
                    peak_max = amplitude
                    peak_pos = i
            start_pos = max(0, peak_pos - self.forward_extension + self.generate_noise(self.noise))
            final_cut_range.append((cr[0] + start_pos, cr[0] + start_pos + self.length))

        return final_cut_range

    def generate_noise(self, noise):
        return random.randint(-noise, noise)
