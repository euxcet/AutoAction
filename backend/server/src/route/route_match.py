from flask import Flask, request, send_file
import base64
from __main__ import app
import file_utils


@app.route("/register_action", methods=["POST"])
def register_action():
    starting_point_base64 = request.form.get("start_point")
    end_point_base64 = request.form.get("end_point")
    starting_point_back_base64 = request.form.get("start_point_back")

    match_path = file_utils.get_match_path()

    with open(os.path.join(match_path, 'start.png'), "wb") as fh:
        fh.write(starting_point_base64).decode('base64')

    with open(os.path.join(match_path, 'end.png'), "wb") as fh:
        fh.write(starting_point_base64).decode('base64')

    with open(os.path.join(match_path, 'start_back.png'), "wb") as fh:
        fh.write(starting_point_base64).decode('base64')