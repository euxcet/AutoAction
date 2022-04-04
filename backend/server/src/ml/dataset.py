import numpy as np
import random
import os
import ml.utils

class Dataset:
    def __init__(self):
        '''
        key: group_id
        value:
            group_name: String
            data: SAMPLE_NUM * LENGTH * CHANNEL
            label: SAMPLE_NUM * record_id
        '''
        self.group_name = dict()
        self.data = dict()
        self.label = dict()

    def get_random_series(self, group_id):
        data = self.data[group_id]
        if data is not None:
            x = random.randint(0, data.shape[0] - 1)
            return data[x]
        else:
            return None

    def get_series(self, group_id, i):
        data = self.data[group_id]
        if data is not None:
            return data[i]
        else:
            return None

    def insert_records(self, records):
        for record in records:
            self.insert_record(record)

    def insert_record(self, record):
        if record.group_id in self.group_name:
            if record.group_name != self.group_name[record.group_id]:
                raise Exception('Inconsistent group names. Existing group name is %s but the new name is %s' % (self.group_name[record.group_id], record.group_name))
        else:
            self.group_name[record.group_id] = record.group_name
            self.data[record.group_id] = None
            self.label[record.group_id] = []

        cur_data = self.data[record.group_id]
        new_data = np.concatenate((record.ex_acc, record.ex_gyro, record.ex_linear), axis=2)
        cur_data = new_data if cur_data is None else np.concatenate((cur_data, new_data), axis=0)
        self.data[record.group_id] = cur_data

        if self.label[record.group_id] is not None:
            for i in range(new_data.shape[0]):
                self.label[record.group_id].append(record.record_id)

        '''
        print(record.group_id, record.record_id, new_data.shape)
        print(self.label[record.group_id])
        exit(0)
        '''

    def export_X_csv(self, dir, data, filename):
        print('Exporting %s...' % (filename), end=' ')
        with open(os.path.join(dir, filename), 'w') as fout:
            fout.write('row_id,series_id,measurement_number,angular_velocity_X,angular_velocity_Y,angular_velocity_Z,linear_acceleration_X,linear_acceleration_Y,linear_acceleration_Z\n')
            count = 0
            for value in data.values():
                if value is None:
                    continue
                for series_id in range(value.shape[0]):
                    for m_number in range(value.shape[1]):
                        # acc0 acc1 acc2 gyro0 gyro1 gyro2 linear0 linear1 linear2
                        v = value[series_id][m_number]
                        fout.write('%d_%d,%d,%d,%f,%f,%f,%f,%f,%f\n' % (count + series_id, m_number, count + series_id, m_number, v[3], v[4], v[5], v[6], v[7], v[8]))
                count += value.shape[0]
        print('\tDone')

    def export_Y_csv(self, dir, data, filename):
        print('Exporting %s...' % (filename), end=' ')
        with open(os.path.join(dir, filename), 'w') as fout:
            fout.write('series_id,group_id,group_name,record_id\n')
            count = 0
            for (key, value) in data.items():
                for i in range(len(value)):
                    fout.write('%d,%d,%s,%s\n' % (count + i, key, self.group_name[key], value[i]))
                count += len(value) 
        print('\tDone')

    def export_csv(self, dir):
        try:
            os.makedirs(dir)
        except:
            pass

        print('\n\nSpliting data...', end=' ')
        train_data, val_data, test_data, train_label, val_label, test_label = ml.utils.split(self.data, self.label, 10, 0, 0)
        print('\tDone')

        self.export_X_csv(dir, train_data, 'X_train.csv')
        self.export_Y_csv(dir, train_label, 'Y_train.csv')
        self.export_X_csv(dir, val_data, 'X_val.csv')
        self.export_Y_csv(dir, val_label, 'Y_val.csv')
        self.export_X_csv(dir, test_data, 'X_test.csv')
        self.export_Y_csv(dir, test_label, 'Y_test.csv')
