import os
import file_utils

class Logger():
    def __init__(self, trainId):
        self.trainId = trainId
        train_path = file_utils.get_train_path(trainId)
        self.fout = open(os.path.join(train_path, "log.txt"), 'w')

    def close(self):
        self.fout.close()

    def flush(self):
        self.fout.flush()

    def log(self, level, content):
        for line in content.split('\n'):
            self.fout.write("[" + level + "]" + line + '\n')
        self.fout.flush()

    def log_info(self, content):
        self.log("I", content)

    def log_error(self, content):
        self.log("E", content)

    def log_debug(self, content):
        self.log("D", content)