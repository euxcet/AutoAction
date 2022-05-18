import os
import json
import shutil
import hashlib
from time import time
from ml.record import Record

DATA_ROOT = os.path.join("..", "data")
DATA_RECORD_ROOT = os.path.join(DATA_ROOT, "record")
DATA_TRAIN_ROOT = os.path.join(DATA_ROOT, "train")
DATA_FILE_ROOT = os.path.join(DATA_ROOT, "file")
DATA_DEX_ROOT = os.path.join(DATA_ROOT, "dex")
DATA_TEMP_ROOT  = os.path.join(DATA_ROOT, "temp")

md5 = dict()

def get_temp_path():
    return DATA_TEMP_ROOT

def get_dex_name_path(name):
    return os.path.join(DATA_DEX_ROOT, name)

def get_dex_user_path(userId, name):
    return os.path.join(get_dex_name_path(name), userId)

def get_dex_path(userId, name, timestamp):
    return os.path.join(get_dex_user_path(userId, name), timestamp)

def get_taskList_path(taskListId):
    return os.path.join(DATA_RECORD_ROOT, taskListId)

def get_task_path(taskListId, taskId):
    return os.path.join(get_taskList_path(taskListId), taskId)

def get_subtask_path(taskListId, taskId, subtaskId):
    return os.path.join(get_task_path(taskListId, taskId), subtaskId)

def get_recordlist_path(taskListId, taskId, subtaskId):
    return os.path.join(get_subtask_path(taskListId, taskId, subtaskId), 'recordlist.txt')

def get_record_path(taskListId, taskId, subtaskId, recordId):
    return os.path.join(get_subtask_path(taskListId, taskId, subtaskId), recordId)

def get_taskList_info_path(taskListId, timestamp = None):
    if timestamp is None or str(timestamp) == "0":
        return os.path.join(get_taskList_path(taskListId), taskListId + ".json")
    return os.path.join(get_taskList_path(taskListId), taskListId + "_" + str(timestamp) + ".json")

def get_task_info_path(taskListid, taskid):
    return os.path.join(get_task_path(taskListid, taskid), taskid + ".json")

def get_subtask_info_path(taskListid, taskid, subtaskId):
    return os.path.join(get_subtask_path(taskListid, taskid, subtaskId), subtaskId + ".json")

def get_train_path(trainId):
    return os.path.join(DATA_TRAIN_ROOT, trainId)

def get_train_info_path(trainId):
    return os.path.join(get_train_path(trainId), trainId + '.json')

def delete_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_json(obj, path):
    with open(path, 'w') as fout:
        json.dump(obj, fout, indent=4)

def load_json(path):
    with open(path, 'r') as fin:
        return json.load(fin)

def load_taskList_info(taskListId, timestamp = None):
    taskList_info_path = get_taskList_info_path(taskListId, timestamp)
    if not os.path.exists(taskList_info_path):
        taskList_info = {"date": "2022.03.14", "description": "Description", "id": taskListId, "task": []}
        save_json(taskList_info, taskList_info_path)
        return taskList_info
    with open(taskList_info_path, 'r') as f:
        data = json.load(f)
        return data

def load_recordlist(taskListId, taskId, subtaskId):
    recordlist_path = get_recordlist_path(taskListId, taskId, subtaskId)
    if not os.path.exists(recordlist_path):
        return []
    recordlist = []
    with open(recordlist_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            recordId = line.strip()
            if recordId.startswith('RD') and recordId not in recordlist:
                recordlist.append(recordId)
    return recordlist

def append_recordlist(taskListId, taskId, subtaskId, recordId):
    recordlist_path = get_recordlist_path(taskListId, taskId, subtaskId)
    with open(recordlist_path, 'a') as f:
        f.write(recordId.strip() + '\n')

def allowed_file(filename):
    return os.path.splitext(filename)[-1] in ['.json', '.mp4', '.bin', '.csv', '.param', '.dex', '.jar']

def save_record_file(file, file_path):
    file.save(file_path)
    '''
    try:
        record_path = "/".join(file_path.split('/')[:-1])
        file_suffix = "_".join((file_path.split('/')[-1]).split('_')[1:])
        if file_suffix.endswith('json'):
            sensor_path = os.path.join(record_path, "Sensor_" + file_suffix)
            timestamp_path = os.path.join(record_path, "Timestamp_" + file_suffix)
            if os.path.exists(sensor_path) and os.path.exists(timestamp_path):
                Record(sensor_path, timestamp_path, do_cut=False)

    except:
        pass
    '''

def save_file(file, file_path):
    file.save(file_path)

def calc_file_md5(file_name):
    m = hashlib.md5()
    with open(file_name, 'rb') as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            m.update(data)
    return m.hexdigest()

def update_md5():
    global md5
    for filename in os.listdir(DATA_FILE_ROOT):
        md5[filename] = calc_file_md5(os.path.join(DATA_FILE_ROOT, filename))

def get_md5(filename):
    global md5
    if filename in md5:
        return md5[filename]
    return ""

    
