import time
import file_utils
import argparse
import os
from multiprocessing import Process

from ml.export import export_csv
from ml.train.train import train_model
from ml.global_vars import GlobalVars

class TrainProcess(Process):
    def __init__(self, train_info_path, train_info, dataset_version='0.2', do_export_csv=True):
        super(TrainProcess, self).__init__()
        self.train_info_path = train_info_path
        self.train_info = train_info
        self.taskListId = train_info['taskListId']
        self.taskIdList = train_info['taskIdList']
        self.trainId = train_info['trainId']
        self.timestamp = train_info['timestamp']
        self.cutter = train_info['cutter']
        self.trainingSession = train_info['trainingSession']
        self.dataset_version = 'dataset_version'
        self.do_export_csv = do_export_csv

    def get_trainId(self):
        return self.trainId

    def interrupt(self):
        file_utils.change_train_status(self.train_info_path, 'Interrupted')

    def run(self):
        ''' The entrance of a training process. Preprocess data and start training.
            Meanwhile record the training status in training info file.
        '''
        # change status from 'Preprocessing' to 'Training'
        file_utils.change_train_status(self.train_info_path, 'Training')

        # config hyperparameters
        motion_sensors = GlobalVars.MOTION_SENSORS
        channel_dim = len(motion_sensors) * 3
        if GlobalVars.FILTER_EN:
            channel_dim *= 4

        config = {
            'channel_dim': channel_dim,
            'sequence_dim': GlobalVars.WINDOW_LENGTH,
            'output_dim': len(self.cutter),
            'lr': 5e-4,
            'epoch': 50,
            'lstm_layer_dim': 1,
            'lstm_hidden_dim': 100,
            'lstm_fc_dim': 40,
        }

        model = None
        for i in range(self.trainingSession):
            # first export csv before training
            if self.do_export_csv:
                export_csv(self.taskListId, self.taskIdList, self.trainId, self.timestamp, self.cutter, self.dataset_version, model=model)

            # start training
            model = train_model(self.trainId, self.timestamp, i, config)

        # change status from 'Training' to 'Done' 
        file_utils.change_train_status(self.train_info_path, 'Done')
        

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default = '../data', help='Root directory of raw data.')
    args = parser.parse_args()
    file_utils.set_data_root(args.data_root)

    # use this code section to debug the training process
    train_info_path = os.path.join(file_utils.DATA_TRAIN_ROOT, 'XT9me9xq7y', 'XT9me9xq7y.json')
    taskListId = 'TL13r912je'
    taskIdList = [
        "TKvx8v7k8l",
        "TK54yquyug",
        "TK7js7kr6d",
        "TKh01oe3tq",
        "TKtdracwmi",
        "TKskodn7oh",
        "TK5codafpv",
        "TKw1r2377b",
        "TKj0838qla",
        "TK4h424fht"
    ]
    trainId = 'XT9me9xq7y'
    timestamp = 142857142857

    new_process = TrainProcess(train_info_path, taskListId, taskIdList, trainId, timestamp, dataset_version='0.1')
    new_process.start()
    new_process.join()

'''