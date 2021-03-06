from ml.export import export_csv
from ml.train.train import train_model
from multiprocessing import Process
import file_utils

class TrainProcess(Process):
    def __init__(self, train_info_path, taskListId, taskIdList, trainId, timestamp):
        super(TrainProcess, self).__init__()
        self.train_info_path = train_info_path
        self.taskListId = taskListId
        self.taskIdList = taskIdList
        self.trainId = trainId
        self.timestamp = timestamp

    def get_trainId(self):
        return self.trainId

    def interrupt(self):
        train_info = file_utils.load_json(self.train_info_path)
        train_info['status'] = 'Interrupted'
        file_utils.save_json(train_info, self.train_info_path)

    def run(self):
        export_csv(self.taskListId, self.taskIdList, self.trainId, self.timestamp)

        train_info = file_utils.load_json(self.train_info_path)
        train_info['status'] = 'Training'
        file_utils.save_json(train_info, self.train_info_path)

        config = dict()
        config['channel_dim'] = 6
        config['sequence_dim'] = 128
        config['layer_dim'] = 1
        config['hidden_dim'] = 512
        config['output_dim'] = 2
        config['lr'] = 0.001
        config['epoch'] = 200
        config['use_cuda'] = True
        train_model(self.trainId, self.timestamp, config)

        train_info = file_utils.load_json(self.train_info_path)
        train_info['status'] = 'Done'
        file_utils.save_json(train_info, self.train_info_path)
