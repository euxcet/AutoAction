from time import time
from flask import Flask, request
from flask_cors import CORS, cross_origin
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import fileUtils
import os


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

saver = ThreadPoolExecutor(max_workers=1)
saver_future_list = []

'''
Data structure:

/data
    /record
        /{tasklistId}
            - {tasklistId}.json
            - {tasklistId}_{timestamp0}.json
            - {tasklistId}_{timestamp1}.json
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


# tasklist related
'''
Name: get_tasklist_history
Method: Get
Form:
    - tasklistId

Respone:
    - List(tasktlist history)
'''
@app.route("/tasklist_history", methods=["GET"])
@cross_origin()
def get_tasklist_history():
    tasklistId = request.form.get("tasklistId")
    tasklist_path = fileUtils.get_tasklist_path(tasklistId)
    response = []
    for file_name in os.listdir(tasklist_path):
        if file_name.startswith("TL") and len(file_name.split('_')) == 2:
            response.append(file_name)
    return {"result": response}

'''
Name: get_tasklist
Method: Get
Form:
    - tasklistId
    - timestamp (Optional)

Respone:
    - tasklist
'''
@app.route("/tasklist", methods=["GET"])
@cross_origin()
def get_tasklist():
    tasklistId = request.form.get("tasklistId")
    timestamp = request.form.get("timestamp")
    tasklist_path = fileUtils.get_tasklist_path(tasklistId)
    print(tasklistId, timestamp)
    tasklist_info_path = fileUtils.get_tasklist_info_path(tasklistId, timestamp)
    if not os.path.exists(tasklist_info_path):
        return {}
    with open(tasklist_info_path, 'r') as f:
        data = json.load(f)
        return data


'''
Name: update_tasklist
Method: Post
Content-Type: multipart/form-data
Form:
    - tasklist
    - timestamp
'''
@app.route("/tasklist", methods=["POST"])
@cross_origin()
def update_tasklist():
    tasklist = json.loads(request.form.get("tasklist"))
    timestamp = int(request.form.get("timestamp"))
    tasklistId = tasklist['id']

    tasklist_info_path = fileUtils.get_tasklist_info_path(tasklistId)
    tasklist_info_timestamp_path = fileUtils.get_tasklist_info_path(tasklistId, timestamp)
    fileUtils.save_json(tasklist, tasklist_info_path)
    fileUtils.save_json(tasklist, tasklist_info_timestamp_path)

    print('tasklist id:', tasklistId)
    for task in tasklist['task']:
        taskId = task['id']
        print('task id:', taskId)
        task_path = fileUtils.get_task_path(tasklistId, taskId)
        task_info_path = fileUtils.get_task_info_path(tasklistId, taskId)
        fileUtils.mkdir(task_path)
        fileUtils.save_json(task, task_info_path)
        print(task_path)
        for subtask in task['subtask']:
            subtaskId = subtask['id']
            subtask_path = fileUtils.get_subtask_path(tasklistId, taskId, subtaskId)
            subtask_info_path = fileUtils.get_subtask_info_path(tasklistId, taskId, subtaskId)
            fileUtils.mkdir(subtask_path)
            fileUtils.save_json(subtask, subtask_info_path)
            print(subtask)
    return ""


# record related
'''
Name: get_record_list
Method: Get
Content-Type: multipart/form-data
Form:
    - tasklistId
    - taskId
    - subtaskId

Response:
    - List(recordId)
'''
@app.route("/record_list", methods=["GET"])
def get_record_list():
    tasklistId = request.form.get("tasklistId")
    taskId = request.form.get("taskId")
    subtaskId = request.form.get("subtaskId")
    subtask_path = fileUtils.get_subtask_path(tasklistId, taskId, subtaskId)
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
    - tasklistId
    - taskId
    - subtaskId
    - recordId
    - timestamp
'''
@app.route("/record", methods=["POST"])
def add_record():
    tasklistId = request.form.get("tasklistId")
    taskId = request.form.get("taskId")
    subtaskId = request.form.get("subtaskId")
    recordId = request.form.get("recordId")
    timestamp = int(request.form.get("timestamp"))
    record_path = fileUtils.get_record_path(tasklistId, taskId, subtaskId, recordId)
    fileUtils.mkdir(record_path)
    fileUtils.save_json({}, os.path.join(record_path, str(timestamp)+ ".json"))
    return {}

'''
Name: delete_record
Method: Post
Content-Type: multipart/form-data
Form:
    - tasklistId
    - taskId
    - subtaskId
    - recordId
'''
@app.route("/record", methods=["DELETE"])
def delete_record():
    tasklistId = request.form.get("tasklistId")
    taskId = request.form.get("taskId")
    subtaskId = request.form.get("subtaskId")
    recordId = request.form.get("recordId")
    record_path = fileUtils.get_record_path(tasklistId, taskId, subtaskId, recordId)
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
    - tasklistId
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
    tasklistId = request.form.get("tasklistId")
    taskId = request.form.get("taskId")
    subtaskId = request.form.get("subtaskId")
    recordId = request.form.get("recordId")
    timestamp = request.form.get("timestamp")

    record_path = fileUtils.get_record_path(tasklistId, taskId, subtaskId, recordId)

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
Content-Type: multipart/form-data
Form:
    - tasklistId
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
Content-Type: multipart/form-data
Form:
    - tasklistId
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
Method: Get
Content-Type: multipart/form-data
Form:
    - tasklistId
    - taskId
    - subtaskId
    - recordId
'''
@app.route("/sample", methods=["DELETE"])
def delete_sample():
    pass


# train related
@app.route("/cutter_type", methods=["GET"])
def get_cutter_type():
    pass


@app.route("/")
@cross_origin()
def hello():
  return "Hello!"

if __name__ == '__main__':
    app.run(port=60010, host="0.0.0.0")
