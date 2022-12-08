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
    parser.add_argument('--export_csv', action='store_true', help='Whether to export training data.')
    args = parser.parse_args()
    file_utils.set_data_root(args.data_root)

    trainId = 'XT9me9xq7y'
    train_info_path = os.path.join(file_utils.DATA_TRAIN_ROOT, trainId, trainId + '.json')
    train_info = file_utils.load_json(train_info_path)

    taskListId = train_info['taskListId']
    taskIdList = train_info['taskIdList']
    timestamp = train_info['timestamp']
    trainingSession = train_info['trainingSession']

    cutter = train_info['cutter']
    for key in cutter:
        for i in range(len(cutter[key])):
            if 'skip_index' in cutter[key][i]['param']:
                si = cutter[key][i]['param']['skip_index']
                if si == 'odd':
                    cutter[key][i]['param']['skip_index'] = lambda x: x % 2 == 1
                elif si == 'even':
                    cutter[key][i]['param']['skip_index'] = lambda x: x % 2 == 0
    print(train_info)

    # new_process = TrainProcess(train_info_path, taskListId, taskIdList, trainId, timestamp, cutter, dataset_version='0.2')
    new_process = TrainProcess(train_info_path, train_info, dataset_version='0.2', do_export_csv=args.export_csv)
    new_process.start()
    new_process.join()
