from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
import os
import nltk
from skin_disease_detector import SkinDiseaseDetector
import cv2
import numpy as np
import threading
import time

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize detector and camera
detector = None
camera = None
last_frame = None
last_prediction = None
prediction_lock = threading.Lock()

class Camera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        return cv2.flip(frame, 1)  # Mirror the frame

def generate_frames():
    global last_frame
    while True:
        if camera is None:
            time.sleep(0.1)
            continue
            
        frame = camera.get_frame()
        if frame is None:
            continue
            
        # Store the frame for prediction
        last_frame = frame.copy()
        
        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global camera
    if camera is None:
        try:
            camera = Camera()
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    return jsonify({'status': 'success'})

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.__del__()
        camera = None
    return jsonify({'status': 'success'})

@app.route('/predict_realtime', methods=['POST'])
def predict_realtime():
    global last_frame, last_prediction, detector
    
    if detector is None:
        return jsonify({'error': 'Model not initialized. Please wait for training to complete.'}), 503
    
    if last_frame is None:
        return jsonify({'error': 'No frame available'}), 400
        
    text_description = request.form.get('description', '')
    
    with prediction_lock:
        try:
            result = detector.predict_realtime(last_frame, text_description)
            last_prediction = result
            # Return all fields including prevention and medicines
            return jsonify({
                'predicted_disease': result['predicted_disease'],
                'confidence': result['confidence'],
                'treatment_recommendation': result['treatment_recommendation'],
                'prevention': result['prevention'],
                'medicines': result['medicines']
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global detector
    
    if detector is None:
        return jsonify({'error': 'Model not initialized. Please wait for training to complete.'}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        text_description = request.form.get('description', '')
        
        try:
            result = detector.predict(filepath, text_description)
            # Return all fields including prevention and medicines
            return jsonify({
                'predicted_disease': result['predicted_disease'],
                'confidence': result['confidence'],
                'treatment_recommendation': result['treatment_recommendation'],
                'prevention': result['prevention'],
                'medicines': result['medicines']
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

def initialize_model():
    global detector
    try:
        print("Initializing model...")
        detector = SkinDiseaseDetector()
        print("Training model...")
        detector.train('data.csv', 'archive (1)/DermNetNZ')
        print("Training completed!")
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        initialize_model()
        app.run(debug=True)
    except Exception as e:
        print(f"Failed to start application: {str(e)}") 