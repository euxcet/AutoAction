import os
import random
import demjson
import matplotlib.pyplot as plt
import numpy as np


'''
Record is parsed from the continuous collected data which may includes several action instances.
'''

class Record:
    def __init__(self, filename, timestamp_filename, group_id = 0, group_name = "", description = "", cutter = None, do_cut = True):
        self.filename = filename
        self.timestamp_filename = timestamp_filename
        self.group_id = group_id
        self.group_name = group_name
        self.description = description
        self.cutter = cutter

        # sensors
        self.acc = []
        self.gyro = []
        self.linear = []
        self.ex_acc = []
        self.ex_gyro = []
        self.ex_linear = []

        self.load_from_file(filename)
        if do_cut:
            self.do_sampling()
            self.cut()

            print(filename, np.array(self.ex_acc).shape)
            self.ex_acc = np.array(self.ex_acc)[:, :, :3]
            self.ex_gyro = np.array(self.ex_gyro)[:, :, :3]
            self.ex_linear = np.array(self.ex_linear)[:, :, :3]

    def cut(self):
        with open(self.timestamp_filename) as fin:
            timestamp = list(map(int, fin.readline().strip()[1:-1].split(',')))
            timestamp.append(timestamp[-1] * 2)
            self.ex_acc, self.ex_gyro, self.ex_linear = self.cutter.cut([self.acc, self.gyro, self.linear], timestamp)

    def do_sampling(self):
        acc_freq = self.calculate_hz(self.acc)
        gyro_freq = self.calculate_hz(self.gyro)
        linear_freq = self.calculate_hz(self.linear)
        print('Accelerometer (Sampling frequency) :', acc_freq, 'Hz')
        print('Gyroscope (Sampling frequency)     :', gyro_freq, 'Hz')
        print('Linear (Sampling frequency)        :', linear_freq, 'Hz')

        if gyro_freq > acc_freq * 0.9 and gyro_freq < acc_freq * 1.1 and linear_freq > acc_freq * 0.9 and linear_freq < acc_freq * 1.1:
            print('No resampling required.')
            return
        
        if linear_freq < acc_freq and linear_freq < gyro_freq:
            print('Downsampling')
            self.acc = self.do_downsampling(self.acc, self.linear)
            self.gyro = self.do_downsampling(self.gyro, self.linear)

        acc_freq = self.calculate_hz(self.acc)
        gyro_freq = self.calculate_hz(self.gyro)
        linear_freq = self.calculate_hz(self.linear)
        print('Accelerometer (Sampling frequency) :', acc_freq, 'Hz')
        print('Gyroscope (Sampling frequency)     :', gyro_freq, 'Hz')
        print('Linear (Sampling frequency)        :', linear_freq, 'Hz')

    def do_downsampling(self, src, refer):
        result = []
        for i in range(len(refer)):
            t = int(1.0 * i / len(refer) * len(src))
            min_dis = 100000000
            min_pos = 0

            for j in range(t - 10, t + 10):
                if j >= 0 and j < len(src):
                    if abs(src[j][3] - refer[i][3]) < min_dis:
                        min_dis = abs(src[j][3] - refer[i][3])
                        min_pos = j
            result.append(src[min_pos])
        return result


    def calculate_hz(self, data):
        return (1e9 * (len(data) - 1)) / (data[-1][3] - data[0][3])


    def load_from_file(self, filename):
        print("Load from file", filename)

        txt_filename = filename[:-4] + 'txt'
        if not os.path.exists(txt_filename):
            print(filename)
            data_json = demjson.decode_file(filename)
            with open(txt_filename, 'w') as fout:
                for t in data_json:
                    fout.write("%f %f %f %f %d " % (t['data'][0] ,t['data'][1], t['data'][2], t['data'][3], t['time']))

        self.data = []
        with open(txt_filename, 'r') as fin:
            line = fin.readline().strip().split(' ')
            for i in range(0, len(line), 5):
                self.data.append([int(float(line[i])), float(line[i + 1]), float(line[i + 2]), float(line[i + 3]), int(line[i + 4])])

        for t in self.data:
            if t[0] == 1: # Accelerometer
                self.acc.append([t[1], t[2], t[3], t[4]])
            elif t[0] == 4: # Gyroscope
                self.gyro.append([t[1], t[2], t[3], t[4]])
            elif t[0] == 10: # Linear acceleration
                self.linear.append([t[1], t[2], t[3], t[4]])
            #elif t[0] == 2: # Magnetic
            #    self.mag.append([t[1], t[2], t[3], t[4]])
        print('Accelerometer (Number of samples) : ', len(self.acc))
        print('Gyroscope (Number of samples)     : ', len(self.gyro))
        print('Linear (Number of samples)        : ', len(self.linear))
        #print('Magnetic: ', len(self.mag))

        #plt.plot([i for i in range(len(self.acc))], [x[0:3] for x in self.acc])
        #plt.show()


    def export_csv(self):
        with open("output/X_train.csv", 'w') as fout:
            fout.write("row_id,series_id,measurement_number,angular_velocity_X,angular_velocity_Y,angular_velocity_Z,linear_acceleration_X,linear_acceleration_Y,linear_acceleration_Z\n")
            for row_id in range(len(self.ex_acc)):
                for series_id in range(len(self.ex_acc[row_id])):
                    acc = self.ex_acc[row_id][series_id]
                    gyro = self.ex_gyro[row_id][series_id]
                    fout.write("%d_%d,%d,%d,%f,%f,%f,%f,%f,%f\n" % (row_id, series_id, row_id, series_id, acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2]))

