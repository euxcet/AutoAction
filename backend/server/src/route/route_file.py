from flask import Flask, request, send_file
from __main__ import app

import file_utils
import os
import json

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