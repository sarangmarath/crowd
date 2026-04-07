# ==========================
# app.py - Crowd AI Flask App
# ==========================

from flask import Flask, render_template, request, send_from_directory
import os
from pipeline.detector import main  # Integrated pipeline function

app = Flask(__name__)

# ---------------------------
# Folders for uploads & outputs
# ---------------------------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# ---------------------------
# DASHBOARD
# ---------------------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

# ---------------------------
# UPLOAD PAGE
# ---------------------------
@app.route("/upload_page")
def upload_page():
    return render_template("upload.html")

# ---------------------------
# UPLOAD VIDEO DETECTION
# ---------------------------
@app.route("/upload", methods=["POST"])
def upload_video():
    file = request.files.get("video")
    
    if not file or file.filename == "":
        return "❌ No file selected"

    input_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    output_filename = "output_" + file.filename
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)

    # Save uploaded video
    file.save(input_path)

    # Run pipeline on uploaded video
    main(
        source=input_path,
        yolo_model_path="models/best.pt",
        umn_model_path="models/umn_abnormal_model.h5",
        output_path=output_path
    )

    # Return result page with processed video
    return render_template("result.html", video=output_filename)

# ---------------------------
# WEBCAM DETECTION
# ---------------------------
@app.route("/webcam")
def webcam():
    """
    Runs live webcam detection.
    Output_path=None means live window only.
    """
    main(
        source=0,  # Default webcam index
        yolo_model_path="models/best.pt",
        umn_model_path="models/umn_abnormal_model.h5",
        output_path=None
    )
    return "🎥 Webcam Closed"

# ---------------------------
# IoT PANEL
# ---------------------------
@app.route("/iot")
def iot():
    return render_template("iot.html")

# ---------------------------
# SERVE OUTPUT VIDEO
# ---------------------------
@app.route("/outputs/<filename>")
def serve_video(filename):
    """
    Serves processed video from outputs folder.
    """
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

# ---------------------------
# RUN FLASK APP
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)