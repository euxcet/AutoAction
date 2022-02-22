from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

'''
Data structure:

/data
    /record
        /{tasklistId}
            - tasklist_{tasklistId}.json
            /{recordId}
                - meta.xml
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
'''
@app.route("/get_tasklist_history", methods=["GET"])
@cross_origin()
def get_tasklsit_history():
    pass


'''
Name: update_tasklist
Method: Post
Content-Type: multipart/form-data
Form:
    - tasklist
    - tasklistId
    - timestamp
'''
@app.route("/update_tasklist", methods=["POST"])
@cross_origin()
def update_tasklist():
    pass


# record related
'''
Name: get_record_list
Method: Get
Content-Type: multipart/form-data
Form:
    - tasklist
    - taskId
    - subtaskId

Response:
    - List(recordId)
'''
@app.route("/get_record_list", methods=["GET"])
def get_record_list():
    pass

'''
Name: add_record
Method: Post
Content-Type: multipart/form-data
Form:
    - tasklist
    - taskId
    - subtaskId
    - recordId
    - timestamp
'''
@app.route("/add_record", methods=["POST"])
def add_record():
    pass


'''
Name: delete_record
Method: Post
Content-Type: multipart/form-data
Form:
    - tasklist
    - taskId
    - subtaskId
    - recordId
'''
@app.route("/delete_record", methods=["POST"])
def delete_record():
    pass



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

Upload files after posting to add_record.
'''
@app.route("/upload_record_file", methods=["POST"])
def upload_file():
    print(request.method)
    if request.method == "POST":
        if "file" not in request.files:
            print("No file part")
            return redirect(request.url)
        file = request.files["file"]
        name = request.form.get("name")
        commit = request.form.get("commit")

        print(file.filename)
        print(name, commit)

        '''
        if file and allowed_file(file.filename):
            filename = name
            print(filename)
            folder = os.path.join(app.config["UPLOAD_FOLDER"], "data")
            if not os.path.exists(folder):
                os.mkdir(folder)

            if os.path.exists(os.path.join(folder, filename)):
                os.remove(os.path.join(folder, filename))

            file.save(os.path.join(folder, filename))

            commit_folder = os.path.join(app.config["UPLOAD_FOLDER"], "commit")
            if not os.path.exists(commit_folder):
                os.mkdir(commit_folder)

            with open(os.path.join(commit_folder, filename + ".txt"), "w") as f:
                f.write(commit)

            return "Succeed"
        '''

    return ""

# sample related
'''
Name: get_sample_number
Method: Get
Content-Type: multipart/form-data
Form:
    - tasklist
    - taskId
    - subtaskId
    - recordId

Response:
    - Number of samples
'''
@app.route("/get_sample_number", methods=["GET"])
def get_sample_number():
    pass

'''
Name: get_sample
Method: Get
Content-Type: multipart/form-data
Form:
    - tasklist
    - taskId
    - subtaskId
    - recordId
    - sampleId

Response:
    - sample
'''
@app.route("/get_sample", methods=["GET"])
def get_sample():
    pass


'''
Name: delete_sample
Method: Get
Content-Type: multipart/form-data
Form:
    - tasklist
    - taskId
    - subtaskId
    - recordId
'''
@app.route("/delete_sample", methods=["POST"])
def delete_sample():
    pass



@app.route("/")
@cross_origin()
def hello():
  return "Hello!"

if __name__ == '__main__':
    app.run(port=60010, host="0.0.0.0")
