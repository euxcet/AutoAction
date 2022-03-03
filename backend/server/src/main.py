from flask import Flask, request, send_file
from flask_cors import CORS, cross_origin
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import fileUtils
import os

from process.cutter.peak_cutter import PeakCutter
from process.cutter.random_cutter import RandomCutter

from process.export import export_csv
from process.train.train import train_model


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

saver = ThreadPoolExecutor(max_workers=1)
saver_future_list = []

trainer = ThreadPoolExecutor(max_workers=1)

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
    for dir in os.listdir(fileUtils.DATA_RECORD_ROOT):
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

    taskList_path = fileUtils.get_taskList_path(taskListId)
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
    return fileUtils.load_taskList_info(taskListId, timestamp)


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

    taskList_info_path = fileUtils.get_taskList_info_path(taskListId)
    taskList_info_timestamp_path = fileUtils.get_taskList_info_path(taskListId, timestamp)
    fileUtils.save_json(taskList, taskList_info_path)
    fileUtils.save_json(taskList, taskList_info_timestamp_path)

    print('taskList id:', taskListId)
    for task in taskList['task']:
        taskId = task['id']
        print('task id:', taskId)
        task_path = fileUtils.get_task_path(taskListId, taskId)
        task_info_path = fileUtils.get_task_info_path(taskListId, taskId)
        fileUtils.mkdir(task_path)
        fileUtils.save_json(task, task_info_path)
        print(task_path)
        for subtask in task['subtask']:
            subtaskId = subtask['id']
            subtask_path = fileUtils.get_subtask_path(taskListId, taskId, subtaskId)
            subtask_info_path = fileUtils.get_subtask_info_path(taskListId, taskId, subtaskId)
            fileUtils.mkdir(subtask_path)
            fileUtils.save_json(subtask, subtask_info_path)
            print(subtask)
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
    subtask_path = fileUtils.get_subtask_path(taskListId, taskId, subtaskId)
    recordIds = []
    for file_name in os.listdir(subtask_path):
        if file_name.startswith("RD"):
            recordIds.apppend(file_name)
    return {"result": recordIds}


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
    record_path = fileUtils.get_record_path(taskListId, taskId, subtaskId, recordId)
    fileUtils.mkdir(record_path)
    fileUtils.save_json({}, os.path.join(record_path, str(timestamp)+ ".json"))
    fileUtils.append_recordlist(taskListId, taskId, subtaskId, recordId)
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
    record_path = fileUtils.get_record_path(taskListId, taskId, subtaskId, recordId)
    fileUtils.delete_dir(record_path)
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
    - taskListId
    - taskId
    - subtaskId
    - recordId
    - timestamp

Upload files after posting to add_record.
'''
@app.route("/record_file", methods=["POST"])
def upload_file():
    file = request.files["file"]
    fileType = request.form.get("fileType")
    taskListId = request.form.get("taskListId")
    taskId = request.form.get("taskId")
    subtaskId = request.form.get("subtaskId")
    recordId = request.form.get("recordId")
    timestamp = request.form.get("timestamp")

    record_path = fileUtils.get_record_path(taskListId, taskId, subtaskId, recordId)
    print("Filename", file.filename)

    if file and fileUtils.allowed_file(file.filename):
        filename = ""
        print("type: ", fileType)
        if fileType == "0": # sensor
            filename = "Sensor_" + str(timestamp) + ".json"
        elif fileType == "1": # sensor
            filename = "Timestamp_" + str(timestamp) + ".json"
        elif fileType == "2": # audio
            filename = "Audio_" + str(timestamp) + ".mp4"
        elif fileType == "3":
            filename = "Video_" + str(timestamp) + ".mp4"

        file_path = os.path.join(record_path, filename)
        #saver_future_list.append(saver.submit(fileUtils.save_record_file, file, file_path))
        fileUtils.save_record_file(file, file_path)
    
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
    for trainId in os.listdir(fileUtils.DATA_TRAIN_ROOT):
        if trainId.startswith('XT'):
            train_info_path = fileUtils.get_train_info_path(trainId)
            response.append(fileUtils.load_json(train_info_path))
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
    trainId = request.form.get("trainId")
    trainName = request.form.get("trainName")
    taskListId = request.form.get("taskListId")
    taskIdList = request.form.get("taskIdList").strip().split(',')
    timestamp = int(request.form.get("timestamp"))
    train_path = fileUtils.get_train_path(trainId)
    fileUtils.mkdir(train_path)
    train_info_path = fileUtils.get_train_info_path(trainId)

    train_info = {
        'name': trainName,
        'id': trainId,
        'taskListId': taskListId,
        'taskIdList': taskIdList,
        'timestamp': timestamp,
        'status': 'Preprocessing'
    }
    fileUtils.save_json(train_info, train_info_path)
    export_csv(taskListId, taskIdList, trainId, timestamp)

    train_info = fileUtils.load_json(train_info_path)
    train_info['status'] = 'Training'
    fileUtils.save_json(train_info, train_info_path)

    train_model(trainId, timestamp, False)

    train_info = fileUtils.load_json(train_info_path)
    train_info['status'] = 'Done'
    fileUtils.save_json(train_info, train_info_path)


    '''
    train_model('9087654321', False)
    taskListId = request.form.get("taskListId")
    taskIds = request.form.get("taskId").strip().split(',')
    timestamp = request.form.get("timestamp")
    export_csv(taskListId, taskIds, timestamp)
    '''
    return {}

@app.route("/download_jar", methods=['GET'])
def download_jar():
    return send_file(os.path.join(fileUtils.DATA_JAR_ROOT, 'classes.dex'))

if __name__ == '__main__':
    app.run(port=60010, host="0.0.0.0")
