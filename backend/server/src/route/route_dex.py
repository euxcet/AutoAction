from flask import Flask, request, send_file
from __main__ import app
import file_utils
import os
import json
import zipfile

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
