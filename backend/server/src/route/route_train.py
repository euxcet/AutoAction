from flask import Flask, request, send_file
from __main__ import app

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
    