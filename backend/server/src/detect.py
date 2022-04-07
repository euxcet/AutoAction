import os
import file_utils
from ml.train.model import LSTMClassifier
import torch
from torch.nn import functional as F
import numpy as np
import onnxruntime
import ncnn

def calc_max_pos(x):
    res = 0
    pos = 0
    for i in range(0, len(x), 6):
        for j in range(3, 6):
            if x[i + j] > res:
                res = x[i + j]
                pos = i // 6
    return pos

def detect(trainId, taskListId, taskId, subtaskId, recordId, class_num):
    train_path = file_utils.DATA_FILE_ROOT
    record_path = file_utils.get_record_path(taskListId, taskId, subtaskId, recordId)
    for f in os.listdir(record_path):
        print(f)

    print('\n\n')
    for f in os.listdir(train_path):
        print(f)

    model = LSTMClassifier(6, 512, 2, class_num, use_cuda=True)
    model.load_state_dict(torch.load(os.path.join(train_path, 'best.pth')))
    model.cuda()

    numpy_data = np.array([1.0 for _ in range(128 * 6)], dtype=np.float32).reshape(1, 128, 6)
    one_input = torch.tensor(numpy_data).float().cuda()
    out = model(one_input)
    print(out)

    session = onnxruntime.InferenceSession(os.path.join(train_path, 'best-sim.onnx'))

    ort_inputs = {session.get_inputs()[0].name: numpy_data}
    ort_outputs = session.run(None, ort_inputs)
    print(ort_outputs)

    net = ncnn.Net()
    net.load_param(os.path.join(train_path, 'best.param'))
    net.load_model(os.path.join(train_path, 'best.bin'))
    in_mat = ncnn.Mat(numpy_data)
    print(in_mat.shape)
    out_mat = ncnn.Mat()
    ex = net.create_extractor()
    ex.input("input", in_mat)
    ex.extract("output", out_mat)
    print(out_mat)


    sensor_filename = ''
    for f in os.listdir(record_path):
        if f.startswith("Sensor") and f.endswith('.txt'):
            sensor_filename = f

    sensor = []
    with open(os.path.join(record_path, sensor_filename), 'r') as fin:
        line = fin.readline().strip().split(' ')
        for i in range(0, len(line), 5):
            sensor.append([int(float(line[i])), float(line[i + 1]), float(line[i + 2]), float(line[i + 3]), int(line[i + 4])])
    
    data_length = 128 * 6
    data_elemsize = 6
    data = [0.0 for _ in range(data_length)]
    lastTimestampGyro = 0
    lastTimestampLinear = 0
    lastKnock = 0
    interval = 9900000
    tot = 0
    for t in sensor:
        if t[0] == 4:
            if t[4] - lastTimestampGyro > interval:
                lastTimestampGyro = t[4]
                for i in range(0, data_length - data_elemsize, data_elemsize):
                    for j in range(0, 3):
                        data[i + j] = data[i + j + data_elemsize]
                    data[data_length - data_elemsize] = t[1]
                    data[data_length - data_elemsize + 1] = t[2]
                    data[data_length - data_elemsize + 2] = t[3]
                input = torch.tensor(np.array(data).reshape(1, 128, 6)).float().cuda()
                out = model(input)
                preds = F.log_softmax(out, dim=1).argmax(dim=1)
                pos = calc_max_pos(data)
                if (pos >= 75 and pos <= 85 and preds[0] == class_num - 1 and t[4] - lastKnock > 1000000000):
                    lastKnock = t[4]
                    print(preds, calc_max_pos(data))
                    tot += 1

        elif t[0] == 10:
            if t[4] - lastTimestampLinear > interval:
                lastTimestampLinear = t[4]
                for i in range(0, data_length - data_elemsize, data_elemsize):
                    for j in range(3, 6):
                        data[i + j] = data[i + j + data_elemsize]
                    data[data_length - data_elemsize + 3] = t[1]
                    data[data_length - data_elemsize + 4] = t[2]
                    data[data_length - data_elemsize + 5] = t[3]
    print(tot)


if __name__ == '__main__':
    class_num = 5
    #trainId = 'XT3sbhnr3n'
    #trainId = 'XTuzpxct73'
    trainId = 'XTd7668a7f'
    #detect(trainId, 'TL13r912je', 'TKj946bp9h', 'ST2c7g7vam', 'RD0xn8gq40', class_num)
    detect(trainId, 'TL13r912je', 'TKtmityjtg', 'STidkjfl81', 'RDlfs01t5i', class_num) #Knock
    #detect(trainId, 'TL13r912je', 'TKjfgtne1v', 'STr00vxc64', 'RD3dcdcn4a', class_num) #Rotate