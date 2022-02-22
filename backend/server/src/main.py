from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


'''
Method: Post
Content-Type: multipart/form-data
Form:
    - file
    - record_id
    - task_id
    - subtask_id
'''
@app.route("/upload", methods=["POST"])
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

@app.route("/")
@cross_origin()
def helloWorld():
  return "Hello, cross-origin-world!"

if __name__ == '__main__':
    app.run(port=60010, host="0.0.0.0")
