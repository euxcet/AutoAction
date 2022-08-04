import time
import file_utils
import argparse
import os
from multiprocessing import Process

from ml.export import export_csv
from ml.train.train import train_model
from ml.global_vars import GlobalVars
from train_process import TrainProcess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default = '../data', help='Root directory of raw data.')
    args = parser.parse_args()
    file_utils.set_data_root(args.data_root)

    # use this code section to debug the training process
    trainId = 'XT9me9xq7y'
    train_info_path = os.path.join(file_utils.DATA_TRAIN_ROOT, trainId, trainId + '.json')
    taskListId = 'TL13r912je'
    taskIdList = [
        "TKkwio3zgi",
        "TKavu21lvb",
        "TKyu7y676q",
        "TKI8tdknq0",
        "TKsf9f7zo2",
        "TKzmggzhdt",
        "TKf774yu2p",
    ]
    timestamp = 142857142857

    file_utils.mkdir(os.path.join(file_utils.DATA_TRAIN_ROOT, trainId))
    file_utils.save_json({}, train_info_path)
    
    new_process = TrainProcess(train_info_path, taskListId, taskIdList, trainId, timestamp, dataset_version='0.2')
    new_process.start()
    new_process.join()

    
