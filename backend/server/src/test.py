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
        "TKknwgwok9",
        "TKdbfxv0c5",
        "TK6k07mgwl",
        "TKb2w0oz3f",
        "TKzq0wdtbg",
    ]
    cutter_type = {
        "TKknwgwok9": ["peak"],
        "TKdbfxv0c5": ["random", "random", "peak"],
        "TK6k07mgwl": ["random", "random", "peak"],
        "TKb2w0oz3f": ["random", "random", "peak"],
        "TKzq0wdtbg": ["peak"],
    }
        #"TKf774yu2p": ["random" for i in range(5)]
    
    timestamp = 142857142857

    file_utils.mkdir(os.path.join(file_utils.DATA_TRAIN_ROOT, trainId))
    file_utils.save_json({}, train_info_path)
    
    new_process = TrainProcess(train_info_path, taskListId, taskIdList, trainId, timestamp, cutter_type, dataset_version='0.2')
    new_process.start()
    new_process.join()
