import os
import numpy as np

from ml.record import Record
from ml.global_vars import GlobalVars


class Dataset:
    def __init__(self):
        ''' Init the data structures.
        attrs:
            self.group_name: group_id:int -> str
            self.data: group_id:int -> np.ndarray, shape =
                (action sample times, time domain length, channels of all sensors)
            self.labels: group_id:int -> list of str, len = sample times
        '''
        self.group_name = dict()
        self.data = dict()
        self.labels = dict()

    def get_random_sample(self, group_id):
        ''' Get a random sample from the dataset.
        args:
            group_id: specifys the data group to get data
        return:
            np.ndarray, shape = (length, channels)
        '''
        data = self.data[group_id]
        if data is not None:
            x = np.random.randint(0, data.shape[0])
            return data[x]
        else: return None

    def get_series(self, group_id, idx):
        ''' Get a sample by index from the dataset.
        args:
            group_id: specifys the data group to get data
            idx: int, sample index in the group 
        return:
            np.ndarray, shape = (length, channels)
        '''
        data = self.data[group_id]
        if data is not None: return data[idx]
        else: return None

    def insert_records(self, records):
        ''' Insert the data of a record list to the dataset.
        '''
        for record in records:
            self.insert_record(record)

    def insert_record(self, record:Record):
        ''' Insert the record data to the dataset.
        '''
        if record.group_id in self.group_name:
            if record.group_name != self.group_name[record.group_id]:
                msg = f'Inconsistent group names. Existing group name is ' \
                    f'{self.group_name[record.group_id]} but the new name is {record.group_name}'
                raise Exception(msg)
        else:
            self.group_name[record.group_id] = record.group_name
            self.data[record.group_id] = None
            self.labels[record.group_id] = []
               
        # create new data for machine learning
        cut_data = record.cut_data
        sensors = GlobalVars.MOTION_SENSORS # motion sensors needed to train the model
        new_data = []
        for sensor in sensors:
            # concatenate x, y, z axes as the three channels
            # first fetch the x, y, z data, each with the shape of (sample times, length in time domain)
            # then create a new axis as the channel axis, and concatencate along this new axis
            sensor_data = cut_data[sensor]
            x, y, z = sensor_data['x'], sensor_data['y'], sensor_data['z']
            new_data.append(np.concatenate([x[:,:,np.newaxis], y[:,:,np.newaxis], z[:,:,np.newaxis]], axis=2))
        # finally concatenate along the new axis to merge different sensor data together
        # shape: (sample times, length in time domain, all channels of all sensors)
        new_data = np.concatenate(new_data, axis=2)
        
        # concatnate the new data to old ones along the 'sample times' axis
        current_data = self.data[record.group_id]
        if current_data is None: current_data = new_data
        else: current_data = np.concatenate([current_data, new_data], axis=0)
        self.data[record.group_id] = current_data
        
        for _ in range(new_data.shape[0]):
            self.labels[record.group_id].append(record.record_id)
            
            
    
    def split(self, train_ratio=0.6, val_ratio=0.1):
        ''' Randomly split self.data and self.labels into train, validation, test sets,
            with ratios specified by train_ratio and val_ratio.
        args:
            train_ratio: the ratio of training samples, in (0, 1), default=0.6
            val_ratio: the ratio of validation samples, in (0, 1), default=0.6
        note:
            1. Call this function after inserting records.
            2. Calc test_ratio = 1.0 - train_ratio - val_ratio, make sure
                (train_ratio + val_ratio) < 1.0 .
        return:
            (train_data, val_data, test_data, train_labels, val_labels, test_labels)
            All data have the same structure as self.data.
            All labels have the same structure as self.labels. 
        '''
        # sanity checks on test_ratio and val_ratio
        assert(0.0 < train_ratio and train_ratio < 1.0)
        assert(0.0 < val_ratio and val_ratio < 1.0)
        assert(train_ratio + val_ratio < 1.0)
        
        # init the data structures
        all_data, all_labels = self.data, self.labels
        train_data, val_data, test_data = dict(), dict(), dict()
        train_labels, val_labels, test_labels = dict(), dict(), dict()
        
        for key, data in all_data.items():
            # first randomly determine train, val, test indices
            # split data into: [ train_data | val_data | test_data ]
            # data.shape: (sample times, time domain length, channels of all sensors)
            labels = np.array(all_labels[key], dtype=str)
            cnt = data.shape[0]
            idxs = np.arange(cnt)
            np.random.shuffle(idxs)
            train_end = int(train_ratio * cnt)
            val_end = train_end + int(val_ratio * cnt)
            train_idxs = idxs[0:train_end]
            val_idxs = idxs[train_end:val_end]
            test_idxs = idxs[val_end:]
            
            # split data and labels by train, val, test idxs
            train_data[key] = data[train_idxs]
            val_data[key] = data[val_idxs]
            test_data[key] = data[test_idxs]
            train_labels[key] = labels[train_idxs]
            val_labels[key] = labels[val_idxs]
            test_labels[key] = labels[test_idxs]
        
        return (train_data, val_data, test_data, train_labels, val_labels, test_labels)


    def export_X_csv(self, dir, data, file_name):
        ''' Export X data in .csv format.
        args:
            dir: train_path, like '../data/train/XTxxxxxxxx/'
            data: train_data, val_data or test_data after spliting,
                with the same structure as self.data
            file_name: .csv file name to be saved
        '''
        print(f'### Exporting {file_name} ...')
        motion_sensors = GlobalVars.MOTION_SENSORS
        # column labels in the .csv file (the first row)
        # sample_id is the index of an complete action sample in data (the first dimension in data)
        # measure_id is the index of a data point within an action sample (the second dimension in data)
        # row_id is a unique row id in the .csv file, in the form of f'{sample_id}_{measure_id}'
        col_labels = ['row_id', 'sample_id', 'measure_id']
        # data points of all motion sensors, in x, y, z
        # like: acc_x, acc_y, acc_z, linear_acc_x, ..., gyro_z.
        for sensor in motion_sensors:
            col_labels.extend([f'{sensor}_x', f'{sensor}_y', f'{sensor}_z'])
        
        with open(os.path.join(dir, file_name), 'w') as fout:
            fout.write(','.join(col_labels) + '\n')
            # sort the keys to make sure X and Y can match in order
            keys = sorted(list(data.keys()))
            cnt = 0
            for key in keys:
                samples = data[key]  # (sample times, length in time domain, number of channels)
                for sample_id, sample in enumerate(samples): # sample: (length, channels)
                    for measure_id, values in enumerate(sample):
                        str_meta = f'{cnt+sample_id}_{measure_id},{cnt+sample_id},{measure_id},'
                        str_values = ','.join([f'{value}' for value in values])
                        fout.write(str_meta + str_values + '\n')
                cnt += len(samples)


    def export_Y_csv(self, dir, labels, file_name):
        ''' Export Y labels in .csv format.
        args:
            dir: train_path, like '../data/train/XTxxxxxxxx/'
            data: train_labels, val_labels or test_labels after spliting,
                with the same structure as self.labels
            file_name: .csv file name to be saved
        '''
        print(f'### Exporting {file_name} ...')
        group_name = self.group_name
        col_labels = ['sample_id', 'group_id', 'group_name', 'record_id']
        with open(os.path.join(dir, file_name), 'w') as fout:
            fout.write(','.join(col_labels) + '\n')
            # sort the keys to make sure X and Y can match in order
            keys = sorted(list(labels.keys()))
            cnt = 0
            for key in keys:
                values = labels[key]
                name = group_name[key]
                for sample_id, value in enumerate(values):
                    fout.write(f'{cnt+sample_id},{key},{name},{value}\n')
                cnt += len(values)


    def export_csv(self, dir):
        ''' Export train, val, test data and labels in .csv format.
        args:
            dir: train_path, like '../data/train/XTxxxxxxxx/'
        '''
        try: os.makedirs(dir)
        except: pass
        
        print(f'### Spliting data ...')
        train_data, val_data, test_data, train_labels, val_labels, test_labels = self.split()

        self.export_X_csv(dir, train_data, 'X_train.csv')
        self.export_Y_csv(dir, train_labels, 'Y_train.csv')
        self.export_X_csv(dir, val_data, 'X_val.csv')
        self.export_Y_csv(dir, val_labels, 'Y_val.csv')
        self.export_X_csv(dir, test_data, 'X_test.csv')
        self.export_Y_csv(dir, test_labels, 'Y_test.csv')
