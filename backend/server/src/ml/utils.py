import random
import numpy as np

def split(data, label, train_size, val_size, test_size):
    total_size = train_size + val_size + test_size
    train_data = dict()
    val_data = dict()
    test_data = dict()
    train_label = dict()
    val_label = dict()
    test_label = dict()

    for key in data.keys():
        train_data[key] = None
        val_data[key] = None
        test_data[key] = None
        train_label[key] = []
        val_label[key] = []
        test_label[key] = []

    for key, value in data.items():
        for i in range(value.shape[0]):
            new_data = value[i][np.newaxis, :]
            new_label = label[key][i]
            t = random.randint(0, total_size - 1)

            if t < train_size:
                train_data[key] = new_data if train_data[key] is None else np.concatenate((train_data[key], new_data), axis=0)
                train_label[key].append(new_label)
            elif t < train_size + val_size:
                val_data[key] = new_data if val_data[key] is None else np.concatenate((val_data[key], new_data), axis=0)
                val_label[key].append(new_label)
            else:
                test_data[key] = new_data if test_data[key] is None else np.concatenate((test_data[key], new_data), axis=0)
                test_label[key].append(new_label)


    return train_data, val_data, test_data, train_label, val_label, test_label