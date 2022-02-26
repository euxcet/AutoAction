import numpy as np
import random
import utils
import os

class Dataset:
    def __init__(self):
        self.group_name = dict()
        self.data = dict()

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

        data = self.data[record.group_id]
        new_data = np.concatenate((record.ex_acc, record.ex_gyro, record.ex_linear), axis=2)

        if data is None:
            data = new_data
        else:
            data = np.concatenate((data, new_data), axis=0)

        self.data[record.group_id] = data

    def export_train_csv(self, dir, data, file_name):
        print('Exporting %s...' % (file_name), end=' ')
        with open(os.path.join(dir, file_name), 'w') as fout:
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

    def export_csv(self, dir):
        try:
            os.makedirs(dir)
        except:
            pass

        print('\n\nSpliting data...', end=' ')
        train_data, val_data, test_data = utils.split(self.data, 10, 0, 0)
        print('\tDone')

        print('\nExporting y_train.csv...', end=' ')
        with open(os.path.join(dir, 'y_train.csv'), 'w') as fout:
            fout.write('series_id,group_id,group_name\n')
            count = 0
            for (key, value) in train_data.items():
                for _ in range(value.shape[0]):
                    fout.write('%d,%d,%s\n' % (count + _, key, self.group_name[key]))
                count += value.shape[0]
        print('\tDone')


        self.export_train_csv(dir, train_data, 'X_train.csv')
        self.export_train_csv(dir, val_data, 'X_val.csv')
        self.export_train_csv(dir, test_data, 'X_test.csv')
