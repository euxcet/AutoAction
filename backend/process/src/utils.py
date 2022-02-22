import random
import numpy as np

def split(data, train_size, val_size, test_size):
    total_size = train_size + val_size + test_size
    train_data = dict()
    val_data = dict()
    test_data = dict()

    for key in data.keys():
        train_data[key] = None
        val_data[key] = None
        test_data[key] = None

    for key, value in data.items():
        for _ in range(value.shape[0]):
            t = random.randint(0, total_size - 1)
            if t < train_size:
                belong = 'train'
                data = train_data[key]
            elif t < train_size + val_size:
                belong = 'val'
                data = val_data[key]
            else:
                belong = 'test'
                data = test_data[key]

            new_data = value[_][np.newaxis, :]

            if data is None:
                data = new_data
            else:
                data = np.concatenate((data, new_data), axis=0)
            
            if belong == 'train':
                train_data[key] = data
            elif belong == 'val':
                val_data[key] = data
            else:
                test_data[key] = data

    return train_data, val_data, test_data