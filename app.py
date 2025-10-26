from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
import os
import shutil
import threading
from datetime import datetime
from processing.video_processor import VideoProcessor
from processing.pointcloud_generator import generate_pointcloud

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "outputs"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max file size
app.config["ALLOWED_EXTENSIONS"] = {"mp4", "avi", "mov", "mkv", "webm"}

socketio = SocketIO(app, cors_allowed_origins="*")


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    fps = request.form.get("fps", 30, type=int)

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify(
            {"error": "Invalid file type. Allowed types: mp4, avi, mov, mkv, webm"}
        ), 400

    # Save uploaded video
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(filename)[0]
    session_id = f"{video_name}_{timestamp}"

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], session_id, filename)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    file.save(video_path)

    return jsonify({"success": True, "session_id": session_id, "filename": filename})


@app.route("/process/<session_id>", methods=["POST"])
def process_video(session_id):
    fps = request.json.get("fps", 30)

    # Start processing in background thread
    thread = threading.Thread(target=process_video_task, args=(session_id, fps))
    thread.start()

    return jsonify({"success": True, "message": "Processing started"})


def process_video_task(session_id, fps):
    try:
        # Setup paths
        upload_dir = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
        video_files = [f for f in os.listdir(upload_dir) if allowed_file(f)]

        if not video_files:
            socketio.emit("error", {"message": "No video file found"})
            return

        video_path = os.path.join(upload_dir, video_files[0])
        temp_frames_dir = os.path.join("processing_temp", session_id)
        output_dir = os.path.join(app.config["OUTPUT_FOLDER"], session_id)

        os.makedirs(temp_frames_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Phase 1: Extract frames (0-40%)
        socketio.emit(
            "progress",
            {
                "session_id": session_id,
                "percentage": 0,
                "status": "Extracting frames from video...",
            },
        )

        processor = VideoProcessor()
        frame_count = processor.extract_frames(
            video_path,
            temp_frames_dir,
            fps,
            progress_callback=lambda p: socketio.emit(
                "progress",
                {
                    "session_id": session_id,
                    "percentage": int(p * 0.4),
                    "status": f"Extracting frames... {int(p)}%",
                },
            ),
        )

        socketio.emit(
            "progress",
            {
                "session_id": session_id,
                "percentage": 40,
                "status": f"Extracted {frame_count} frames. Generating pointcloud...",
            },
        )

        # Phase 2: Generate pointcloud (40-90%)
        generate_pointcloud(
            temp_frames_dir,
            progress_callback=lambda p: socketio.emit(
                "progress",
                {
                    "session_id": session_id,
                    "percentage": int(40 + p * 0.5),
                    "status": f"Generating pointcloud... {int(p)}%",
                },
            ),
        )

        socketio.emit(
            "progress",
            {
                "session_id": session_id,
                "percentage": 90,
                "status": "Packaging output files...",
            },
        )

        # Phase 3: Package output (90-100%)
        # Create output structure
        frames_output_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_output_dir, exist_ok=True)

        # Copy original video
        shutil.copy(
            video_path,
            os.path.join(
                output_dir, f"original_video{os.path.splitext(video_files[0])[1]}"
            ),
        )

        # Copy frames
        for frame_file in os.listdir(temp_frames_dir):
            if frame_file.endswith((".jpg", ".png")):
                shutil.copy(
                    os.path.join(temp_frames_dir, frame_file),
                    os.path.join(frames_output_dir, frame_file),
                )

        # Copy pointcloud file
        if "/results" and os.path.exists("/results"):
            shutil.copy(
                "/results", os.path.join(output_dir, os.path.basename("/results"))
            )

        # Create zip file
        zip_path = shutil.make_archive(
            os.path.join(app.config["OUTPUT_FOLDER"], f"{session_id}"),
            "zip",
            output_dir,
        )

        # Cleanup temp files
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        shutil.rmtree(upload_dir, ignore_errors=True)

        socketio.emit(
            "progress",
            {
                "session_id": session_id,
                "percentage": 100,
                "status": "Processing complete!",
                "download_url": f"/download/{session_id}",
            },
        )

    except Exception as e:
        socketio.emit("error", {"session_id": session_id, "message": str(e)})


@app.route("/download/<session_id>")
def download(session_id):
    zip_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{session_id}.zip")

    if not os.path.exists(zip_path):
        return jsonify({"error": "File not found"}), 404

    return send_file(zip_path, as_attachment=True, download_name=f"{session_id}.zip")


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("processing_temp", exist_ok=True)

    socketio.run(app, debug=True, host="0.0.0.0", port=5001, allow_unsafe_werkzeug=True)
