from flask import Flask, request, send_file
from __main__ import app

# TODO

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