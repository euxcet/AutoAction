from hashlib import new
from re import sub
from time import time
from flask import Flask, request, send_file
from flask_cors import CORS, cross_origin
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import file_utils
import os

import zipfile

from ml.cutter.peak_cutter import PeakCutter
from ml.cutter.random_cutter import RandomCutter

from train_process import TrainProcess

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

saver = ThreadPoolExecutor(max_workers=1)
saver_future_list = []

trainer = ThreadPoolExecutor(max_workers=1)
train_processes = []

file_utils.mkdir("../data/record/TL13r912je")
if not os.path.exists("../data/file/config.json"):
    config = {
        "context": [
            {
                "builtInContext": "Informational",
                "sensorType": ["ACCESSIBILITY", "BROADCAST"],
                "integerParamKey": [],
                "integerParamValue": [],
                "longParamKey": [],
                "longParamValue": [],
                "floatParamKey": [],
                "floatParamValue": [],
                "booleanParamKey": [],
                "booleanParamValue": []
            },
            {
                "builtInContext": "Config",
                "sensorType": ["ACCESSIBILITY", "BROADCAST"],
                "integerParamKey": [],
                "integerParamValue": [],
                "longParamKey": [],
                "longParamValue": [],
                "floatParamKey": [],
                "floatParamValue": [],
                "booleanParamKey": [],
                "booleanParamValue": []
            }
        ],
        "action": [
            {
                "builtInAction": "TapTap",
                "sensorType": ["IMU"],
                "integerParamKey": ["SeqLength"],
                "integerParamValue": [50],
                "longParamKey": [],
                "longParamValue": [],
                "floatParamKey": [],
                "floatParamValue": [],
                "booleanParamKey": [],
                "booleanParamValue": []
            },
            {
                "builtInAction": "TopTap",
                "sensorType": ["IMU"],
                "integerParamKey": ["SeqLength"],
                "integerParamValue": [25],
                "longParamKey": [],
                "longParamValue": [],
                "floatParamKey": [],
                "floatParamValue": [],
                "booleanParamKey": [],
                "booleanParamValue": []
            }
        ],
        "listenedSystemActions": [
            "android.intent.action.AIRPLANE_MODE",
            "android.intent.action.SCREEN_OFF",
            "android.intent.action.SCREEN_ON",
            "android.bluetooth.device.action.ACL_CONNECTED",
            "android.intent.action.APPLICATION_RESTRICTIONS_CHANGED",
            "android.intent.action.BATTERY_LOW",
            "android.intent.action.BATTERY_OKAY",
            "android.intent.action.BOOT_COMPLETED",
            "android.intent.action.CONFIGURATION_CHANGED",
            "android.intent.action.DOCK_EVENT",
            "android.intent.action.DREAMING_STARTED",
            "android.intent.action.DREAMING_STOPPED",
            "android.intent.action.EXTERNAL_APPLICATIONS_AVAILABLE",
            "android.intent.action.EXTERNAL_APPLICATIONS_UNAVAILABLE",
            "android.intent.action.HEADSET_PLUG",
            "android.intent.action.INPUT_METHOD_CHANGED",
            "android.intent.action.LOCALE_CHANGED",
            "android.intent.action.LOCKED_BOOT_COMPLETED",
            "android.intent.action.MEDIA_BAD_REMOVAL",
            "android.intent.action.MEDIA_BUTTON",
            "android.intent.action.MEDIA_CHECKING",
            "android.intent.action.MEDIA_EJECT",
            "android.intent.action.MEDIA_MOUNTED",
            "android.intent.action.MEDIA_NOFS",
            "android.intent.action.MEDIA_REMOVED",
            "android.intent.action.MEDIA_SCANNER_FINISHED",
            "android.intent.action.MEDIA_SCANNER_STARTED",
            "android.intent.action.MEDIA_SHARED",
            "android.intent.action.MEDIA_UNMOUNTABLE",
            "android.intent.action.MEDIA_UNMOUNTED",
            "android.intent.action.MY_PACKAGE_REPLACED",
            "android.intent.action.PACKAGES_SUSPENDED",
            "android.intent.action.PACKAGES_UNSUSPENDED",
            "android.intent.action.PACKAGE_ADDED",
            "android.intent.action.PACKAGE_CHANGED",
            "android.intent.action.PACKAGE_DATA_CLEARED",
            "android.intent.action.PACKAGE_FIRST_LAUNCH",
            "android.intent.action.PACKAGE_FULLY_REMOVED",
            "android.intent.action.PACKAGE_NEEDS_VERIFICATION",
            "android.intent.action.PACKAGE_REMOVED",
            "android.intent.action.PACKAGE_REPLACED",
            "android.intent.action.PACKAGE_RESTARTED",
            "android.intent.action.PACKAGE_VERIFIED",
            "android.intent.action.ACTION_POWER_CONNECTED",
            "android.intent.action.ACTION_POWER_DISCONNECTED",
            "android.intent.action.PROVIDER_CHANGED",
            "android.intent.action.REBOOT",
            "android.intent.action.ACTION_SHUTDOWN",
            "android.intent.action.TIMEZONE_CHANGED",
            "android.intent.action.TIME_SET",
            "android.intent.action.UID_REMOVED",
            "android.intent.action.USER_BACKGROUND",
            "android.intent.action.USER_FOREGROUND",
            "android.intent.action.USER_PRESENT",
            "android.intent.action.USER_UNLOCKED",
            "android.bluetooth.device.action.ACL_DISCONNECT_REQUESTED",
            "android.bluetooth.device.action.ACL_DISCONNECTED",
            "android.net.wifi.STATE_CHANGE",
            "android.net.wifi.WIFI_STATE_CHANGED"
        ],
        "listenedSystemURIs": [
            "content://settings/system",
            "content://settings/global"
        ],
        "overrideSystemActions": False,
        "overrideSystemURIs": False
    }

    file_utils.save_json(config, "../data/file/config.json")


'''
Data structure:

/data
    /record
        /{taskListId}
            - {taskListId}.json
            - {taskListId}_{timestamp0}.json
            - {taskListId}_{timestamp1}.json
            - ...
            /{taskId}
                - {taskId}.json
                /{subtaskId}
                    - {subtaskId}.json
                    /{recordId}
                        - sensor_{recordId}.json
                        - timestamp_{recordId}.json
                        - audio_{recordId}.mp4
                        - video_{recordId}.mp4
                        - sample_{recordId}.csv

    /dex
        /userId
            /name
                /timestamp
                    - {}.bin
'''


# taskList related
'''
Name: get_all_taskList
Method: Get
Respone:
    - List(tasktListId)
'''
@app.route("/all_taskList", methods=["GET"])
@cross_origin()
def get_all_taskList():
    response = []
    for dir in os.listdir(file_utils.DATA_RECORD_ROOT):
        if dir.startswith("TL"):
            response.append(dir)
    return {"result": response}

'''
Name: get_taskList_history
Method: Get
Form:
    - taskListId

Respone:
    - List(tasktlist history)
'''
@app.route("/taskList_history", methods=["GET"])
@cross_origin()
def get_taskList_history():
    taskListId = request.args.get("taskListId")

    taskList_path = file_utils.get_taskList_path(taskListId)
    response = []
    for file_name in os.listdir(taskList_path):
        if file_name.startswith("TL") and len(file_name.split('_')) == 2:
            response.append(file_name)
    return {"result": response}

'''
Name: get_taskList
Method: Get
Form:
    - taskListId
    - timestamp (Optional)

Respone:
    - taskList
'''
@app.route("/taskList", methods=["GET"])
@cross_origin()
def get_taskList():
    taskListId = request.args.get("taskListId")
    timestamp = request.args.get("timestamp")
    print(taskListId, timestamp)
    return file_utils.load_taskList_info(taskListId, timestamp)


'''
Name: update_taskList
Method: Post
Content-Type: multipart/form-data
Form:
    - taskList
    - timestamp
'''
@app.route("/taskList", methods=["POST"])
@cross_origin()
def update_taskList():
    taskList = json.loads(request.form.get("taskList"))
    timestamp = int(request.form.get("timestamp"))
    taskListId = taskList['id']

    taskList_info_path = file_utils.get_taskList_info_path(taskListId)
    taskList_info_timestamp_path = file_utils.get_taskList_info_path(taskListId, timestamp)
    file_utils.save_json(taskList, taskList_info_path)
    file_utils.save_json(taskList, taskList_info_timestamp_path)

    print('taskList id:', taskListId)
    for task in taskList['task']:
        taskId = task['id']
        print('task id:', taskId)
        task_path = file_utils.get_task_path(taskListId, taskId)
        task_info_path = file_utils.get_task_info_path(taskListId, taskId)
        file_utils.mkdir(task_path)
        file_utils.save_json(task, task_info_path)
        for subtask in task['subtask']:
            subtaskId = subtask['id']
            subtask_path = file_utils.get_subtask_path(taskListId, taskId, subtaskId)
            subtask_info_path = file_utils.get_subtask_info_path(taskListId, taskId, subtaskId)
            file_utils.mkdir(subtask_path)
            file_utils.save_json(subtask, subtask_info_path)
    return ""


# record related
'''
Name: get_record_list
Method: Get
Form:
    - taskListId
    - taskId
    - subtaskId

Response:
    - List(recordId)
'''
@app.route("/record_list", methods=["GET"])
def get_record_list():
    taskListId = request.args.get("taskListId")
    taskId = request.args.get("taskId")
    subtaskId = request.args.get("subtaskId")

    if taskListId is None:
        return {}

    taskList = file_utils.load_taskList_info(taskListId, 0)
    records = []
    for task in taskList['task']:
        c_taskId = task['id']
        if taskId is not None and taskId != "0" and c_taskId != task:
            continue
        for subtask in task['subtask']:
            c_subtaskId = subtask['id']
            if subtaskId is not None and subtaskId != "0" and c_subtaskId != subtaskId:
                continue
            recordlist_path = file_utils.get_recordlist_path(taskListId, c_taskId, c_subtaskId)
            if os.path.exists(recordlist_path):
                with open(recordlist_path, 'r') as fin:
                    lines = fin.readlines()
                    for recordId in lines:
                        recordId = recordId.strip()
                        record_path = file_utils.get_record_path(taskListId, c_taskId, c_subtaskId, recordId)
                        timestamp = 0
                        if os.path.exists(record_path):
                            for filename in os.listdir(record_path):
                                if len(filename.split('_')) == 1 and filename.endswith('.json'):
                                    timestamp = int(filename.split('.')[0])

                        records.append({
                            'taskListId': taskListId,
                            'taskId': c_taskId,
                            'subtaskId': c_subtaskId,
                            'recordId': recordId.strip(),
                            'timestamp': timestamp
                        })

                        records.sort(key = lambda x: -x['timestamp'])
    return {'recordList': records}

'''
Name: add_record
Method: Post
Content-Type: multipart/form-data
Form:
    - taskListId
    - taskId
    - subtaskId
    - recordId
    - timestamp
'''
@app.route("/record", methods=["POST"])
def add_record():
    taskListId = request.form.get("taskListId")
    taskId = request.form.get("taskId")
    subtaskId = request.form.get("subtaskId")
    recordId = request.form.get("recordId")
    timestamp = int(request.form.get("timestamp"))
    record_path = file_utils.get_record_path(taskListId, taskId, subtaskId, recordId)
    file_utils.mkdir(record_path)
    file_utils.save_json({}, os.path.join(record_path, str(timestamp)+ ".json"))
    file_utils.append_recordlist(taskListId, taskId, subtaskId, recordId)
    return {}

'''
Name: delete_record
Method: Delete
Content-Type: multipart/form-data
Form:
    - taskListId
    - taskId
    - subtaskId
    - recordId
'''
@app.route("/record", methods=["DELETE"])
def delete_record():
    taskListId = request.form.get("taskListId")
    taskId = request.form.get("taskId")
    subtaskId = request.form.get("subtaskId")
    recordId = request.form.get("recordId")
    record_path = file_utils.get_record_path(taskListId, taskId, subtaskId, recordId)
    file_utils.delete_dir(record_path)
    return {}

def get_filetype_prefix(fileType):
    if fileType == "0": # sensor
        return "Sensor_"
    elif fileType == "1": # sensor
        return "Timestamp_"
    elif fileType == "2": # audio
        return "Audio_"
    elif fileType == "3": # video
        return "Video_"
    elif fileType == "4": # sensor bin
        return "SensorBin_"

def get_filetype_ext(fileType):
    if fileType == "0": # sensor
        return ".json"
    elif fileType == "1": # sensor
        return ".json"
    elif fileType == "2": # audio
        return ".mp4"
    elif fileType == "3": # video
        return ".mp4"
    elif fileType == "4": # sensor bin
        return ".bin"

'''
Name: download_record_file
Method: Post
Content-Type: multipart/form-data
Form:
    - taskListId
    - taskId
    - subtaskId
    - recordId
    - fileType
        - 0 sensor json
        - 1 timestamp json
        - 2 audio mp4
        - 3 video mp4
        - 4 sensor bin

'''
@app.route("/record_file", methods=['GET'])
def download_record():
    taskListId = request.args.get("taskListId")
    taskId = request.args.get("taskId")
    subtaskId = request.args.get("subtaskId")
    recordId = request.args.get("recordId")
    fileType = request.args.get("fileType")

    prefix = get_filetype_prefix(fileType)
    record_path = file_utils.get_record_path(taskListId, taskId, subtaskId, recordId)
    if os.path.exists(record_path):
        for filename in os.listdir(record_path):
            if filename.startswith(prefix):
                return send_file(os.path.join(record_path, filename))

    return {}

'''
Name: update_record_file
Method: Post
Content-Type: multipart/form-data
Form:
    - file
    - fileType
        - 0 sensor json
        - 1 timestamp json
        - 2 audio mp4
        - 3 video mp4
        - 4 sensor bin
    - taskListId
    - taskId
    - subtaskId
    - recordId
    - timestamp

Upload files after posting to record.
'''
@app.route("/record_file", methods=["POST"])
def upload_record_file():
    file = request.files["file"]
    fileType = request.form.get("fileType")
    taskListId = request.form.get("taskListId")
    taskId = request.form.get("taskId")
    subtaskId = request.form.get("subtaskId")
    recordId = request.form.get("recordId")
    timestamp = request.form.get("timestamp")

    record_path = file_utils.get_record_path(taskListId, taskId, subtaskId, recordId)
    print("Filename", file.filename)

    if file and file_utils.allowed_file(file.filename):
        filename = ""
        print("type: ", fileType)
        prefix, ext = get_filetype_prefix(fileType), get_filetype_ext(fileType)
        filename = prefix + str(timestamp) + ext
        file_path = os.path.join(record_path, filename)
        file_utils.save_record_file(file, file_path)
    
    return {}


'''
Name: upload_collected_data
Method: Post
Content-Type: multipart/form-data
Form:
    - file
    - fileType
        - 0 sensor bin
    - userId
    - name
    - commit
    - timestamp


    /dex
        /name
            /userId
                /timestamp
                    - {}.bin
'''

@app.route("/collected_data", methods=['POST'])
def upload_collected_data():
    file = request.files["file"]
    print(file.filename)
    meta = json.loads(request.form.get("meta"))
    print(meta)

    if file.filename[-4:] == '.zip':
        temp_path = file_utils.get_temp_path()
        file_path = os.path.join(temp_path, file.filename)
        file_utils.mkdir(temp_path)
        file_utils.save_file(file, file_path)

        file_zip = zipfile.ZipFile(file_path, 'r')
        for name in file_zip.namelist():
            print(name)
            meta_ = None
            for m in meta:
                if m['file'] == name:
                    meta_ = m
            print(meta_)
            if meta_ is not None:
                path = file_utils.get_dex_path(meta_['userId'], meta_['name'], str(meta_['timestamp']))
                file_utils.mkdir(path)
                file_zip.extract(meta_['file'], path)
                with open(os.path.join(path, meta_['file'] + '.meta'), 'w') as fout:
                    fout.write(json.dumps(meta_))
        file_zip.close()
        os.remove(file_path)
    else:
        path = file_utils.get_dex_path(meta[0]['userId'], meta[0]['name'], str(meta[0]['timestamp']))
        file_utils.mkdir(path)
        file_path = os.path.join(path, meta[0]['file'])
        file_utils.save_file(file, file_path)
        with open(os.path.join(path, meta[0]['file'] + '.meta'), 'w') as fout:
            fout.write(json.dumps(meta[0]))


    '''
    fileType = request.form.get("fileType")
    userId = request.form.get("userId")
    name = request.form.get("name")
    timestamp = request.form.get("timestamp")
    path = file_utils.get_dex_path(userId, name, timestamp)
    file_path = os.path.join(path, file.filename)
    print(f"saving file: {file_path}")
    file_utils.mkdir(path)
    file_utils.save_file(file, file_path)
    '''
    return {}


# sample related
'''
Name: get_sample_number
Method: Get
Form:
    - taskListId
    - taskId
    - subtaskId
    - recordId

Response:
    - Number of samples
'''
@app.route("/sample_number", methods=["GET"])
def get_sample_number():
    pass

'''
Name: get_sample
Method: Get
Form:
    - taskListId
    - taskId
    - subtaskId
    - recordId
    - sampleId

Response:
    - sample
'''
@app.route("/sample", methods=["GET"])
def get_sample():
    pass


'''
Name: delete_sample
Method: Delete
Content-Type: multipart/form-data
Form:
    - taskListId
    - taskId
    - subtaskId
    - recordId
'''
@app.route("/sample", methods=["DELETE"])
def delete_sample():
    pass


# train related
'''
Name: get_cutter_type
Method: Get
Response: 
    - cutter_type
'''
@app.route("/cutter_type", methods=["GET"])
def get_cutter_type():
    response = []
    cutters = [PeakCutter(0), RandomCutter()]
    for cutter in cutters:
        response.append(cutter.to_json())
    return {"result": response}

'''
Name: get_train_list
Method: Get
Response:
    - List(trainId)
'''
@app.route("/train_list", methods=["GET"])
def get_train_list():
    response = []
    for trainId in os.listdir(file_utils.DATA_TRAIN_ROOT):
        if trainId.startswith('XT'):
            train_info_path = file_utils.get_train_info_path(trainId)
            response.append(file_utils.load_json(train_info_path))
    response.sort(key=lambda x: -x['timestamp'])
    return {"trainList": response}

'''
Name: start_train
Method: Post
Content-Type: multipart/form-data
Form:
    - trainId
    - trainName
    - taskListId
    - taskIdList  List(taskId)
    - timestamp
'''
@app.route("/train", methods=["POST"])
def start_train():
    global train_processes
    trainId = request.form.get("trainId")
    trainName = request.form.get("trainName")
    taskListId = request.form.get("taskListId")
    taskIdList = request.form.get("taskIdList").strip().split(',')
    timestamp = int(request.form.get("timestamp"))
    train_path = file_utils.get_train_path(trainId)
    file_utils.mkdir(train_path)
    train_info_path = file_utils.get_train_info_path(trainId)

    train_info = {
        'name': trainName,
        'id': trainId,
        'taskListId': taskListId,
        'taskIdList': taskIdList,
        'timestamp': timestamp,
        'status': 'Preprocessing'
    }
    file_utils.save_json(train_info, train_info_path)

    new_process = TrainProcess(train_info_path, taskListId, taskIdList, trainId, timestamp)
    train_processes.append(new_process)
    new_process.start()
    return {}

'''
Name: stop_train
Method: Delete
Content-Type: multipart/form-data
Form:
    - trainId
'''
@app.route("/train", methods=["DELETE"])
def stop_train():
    global train_processes
    trainId = request.form.get("trainId")
    for process in train_processes:
        if process is not None and process.get_trainId() == trainId:
            process.interrupt()
            process.terminate()
            train_processes.remove(process)
    train_processes = list(filter(None, train_processes))
    return {}
    

'''
Name: download_file
Method: Post
Content-Type: multipart/form-data
Form:
    - filename
'''
@app.route("/file", methods=['GET'])
def download_file():
    filename = request.args.get("filename")
    return send_file(os.path.join(file_utils.DATA_FILE_ROOT, filename))
    
'''
Name: update_file
Method: Post
Content-Type: multipart/form-data
Form:
    - file
'''
@app.route("/file", methods=["POST"])
def upload_file():
    file = request.files["file"]
    if file:
        file_utils.save_file(file, os.path.join(file_utils.DATA_FILE_ROOT, file.filename))
        file_utils.update_md5()
    
    return {}

'''
Name: get_md5
Method: Post
Content-Type: multipart/form-data
Form:
    - filename
'''
@app.route("/md5", methods=['GET'])
def get_md5():
    filenames = request.args.get("filename").strip().split(',')
    result = ""
    for filename in filenames:
        if filename != '':
            result += file_utils.get_md5(filename) + ","
    return result[:-1]

'''
Name: update_md5
Method: Post
Content-Type: multipart/form-data
Form:
'''
@app.route("/md5", methods=['POST'])
def update_md5():
    file_utils.update_md5()
    return {}
        


if __name__ == '__main__':
    update_md5()
    app.run(port=6125, host="0.0.0.0")
