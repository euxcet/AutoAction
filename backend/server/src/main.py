from flask import Flask, request, send_file
from flask_cors import CORS, cross_origin
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import file_utils
import os

app = Flask(__name__)
# CORS: A Flask extension for handling Cross Origin Resource Sharing
# (CORS), making cross-origin AJAX possible.
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# routes
from route.route_record import *
from route.route_tasklist import *
from route.route_train import *
from route.route_file import *
from route.route_dex import *

# multi-thread saver
saver = ThreadPoolExecutor(max_workers=1)
saver_future_list = []


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


if __name__ == '__main__':
    file_utils.create_default_files()
    update_md5()
    app.run(port=6125, host="0.0.0.0")
