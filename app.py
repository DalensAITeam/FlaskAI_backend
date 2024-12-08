from flask import Flask, Response, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
from ultralytics import YOLO
import base64
import threading
from model import Animal


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

ALLOWED_VIDEO_EXTENSIONS = {'mp4'}
def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

# Load YOLOv8 model
model = YOLO("yolov8n.pt")


class animal_model:
    """
    Class for chicken detection
    """
    def __init__(self) -> None:
        pass


    def process_video( self,file_path,animal_name):
         animal = Animal()
         for frame, threat_state, animal_number in animal.run( file_path,animal_name):

                detection_text =  [f"Animal_Threat_State: {threat_state}  , Animal_number: {animal_number}"]
                socketio.emit('detection_update', {'text': detection_text})  # Send detection data to frontend

                # Convert frame to base64
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')

                # Emit the frame to the frontend via WebSocket
                socketio.emit('frame_update', frame_base64)

                # Sleep briefly to simulate real-time streaming
                socketio.sleep(0.1)
                # print(threat_state)
@app.route('/')
def index():
    return {"Response":"welcome"}

# Route to handle video upload and processing
@app.route('/video_feed', methods=["POST"])
def video_feed():
    if request.method == 'POST':
        input_video = request.files['file']
        file_path = './videos/' + input_video.filename
        animal_name = request.form.get('animal_name')
        input_video.save(file_path)

        model = animal_model()

        # Start a new thread to process the video in the background
        threading.Thread(target=model.process_video, args=(file_path,animal_name)).start()

        return jsonify({"response": "Video processing started"}), 200


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=7017, debug=True,allow_unsafe_werkzeug=True)
