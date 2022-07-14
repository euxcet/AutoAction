from flask import Flask, request, send_file
from __main__ import app

import file_utils
import os
import json

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
    for task in taskList['tasks']:
        c_taskId = task['id']
        if taskId is not None and taskId != "0" and c_taskId != task:
            continue
        for subtask in task['subtasks']:
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
    if fileType == '0': return 'Timestamp_'
    elif fileType == '1': return 'Motion_'
    elif fileType == '2': return 'Light_'
    elif fileType == '3': return 'Audio_'
    elif fileType == '4': return 'Video_'

def get_filetype_ext(fileType):
    if fileType == '0': return '.json'
    elif fileType == '1': return '.bin'
    elif fileType == '2': return '.bin'
    elif fileType == '3': return '.mp4'
    elif fileType == '4': return '.mp4'
    
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
        - 0 timestamp json
        - 1 motion bin
        - 2 light bin
        - 3 audio mp4
        - 4 video mp4
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
    
    if file and file_utils.allowed_file(file.filename):
        prefix, ext = get_filetype_prefix(fileType), get_filetype_ext(fileType)
        filename = prefix + str(timestamp) + ext
        file_path = os.path.join(record_path, filename)
        file_utils.save_record_file(file, file_path)
    
    return {}
