import os
import json
import shutil
from time import time
from process.record import Record

DATA_ROOT = '../data/'
DATA_RECORD_ROOT = os.path.join(DATA_ROOT, "record")
DATA_TRAIN_ROOT = os.path.join(DATA_ROOT, "train")

def get_tasklist_path(tasklistId):
    return os.path.join(DATA_RECORD_ROOT, tasklistId)

def get_task_path(tasklistId, taskId):
    return os.path.join(get_tasklist_path(tasklistId), taskId)

def get_subtask_path(tasklistId, taskId, subtaskId):
    return os.path.join(get_task_path(tasklistId, taskId), subtaskId)

def get_recordlist_path(tasklistId, taskId, subtaskId):
    return os.path.join(get_subtask_path(tasklistId, taskId, subtaskId), 'recordlist.txt')

def get_record_path(tasklistId, taskId, subtaskId, recordId):
    return os.path.join(get_subtask_path(tasklistId, taskId, subtaskId), recordId)

def get_tasklist_info_path(tasklistId, timestamp = None):
    if timestamp is None or timestamp == 0:
        return os.path.join(get_tasklist_path(tasklistId), tasklistId + ".json")
    return os.path.join(get_tasklist_path(tasklistId), tasklistId + "_" + str(timestamp) + ".json")

def get_task_info_path(tasklistid, taskid):
    return os.path.join(get_task_path(tasklistid, taskid), taskid + ".json")

def get_subtask_info_path(tasklistid, taskid, subtaskId):
    return os.path.join(get_subtask_path(tasklistid, taskid, subtaskId), subtaskId + ".json")

def get_train_path(timestamp):
    return os.path.join(DATA_TRAIN_ROOT, str(timestamp))

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

def load_tasklist_info(tasklistId, timestamp = None):
    tasklist_info_path = get_tasklist_info_path(tasklistId, timestamp)
    if not os.path.exists(tasklist_info_path):
        return {}
    with open(tasklist_info_path, 'r') as f:
        data = json.load(f)
        return data

def load_recordlist(tasklistId, taskId, subtaskId):
    recordlist_path = get_recordlist_path(tasklistId, taskId, subtaskId)
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

def append_recordlist(tasklistId, taskId, subtaskId, recordId):
    recordlist_path = get_recordlist_path(tasklistId, taskId, subtaskId)
    with open(recordlist_path, 'a') as f:
        f.write(recordId.strip() + '\n')

def allowed_file(filename):
    return os.path.splitext(filename)[-1] in ['.json', '.mp4']

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