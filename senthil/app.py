from flask import Flask, request, render_template, redirect, url_for, Response
from pymongo import MongoClient
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from datetime import datetime
import os
from dotenv import load_dotenv
import cv2
from fall_detection import FallDetector

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure Twilio
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
RECIPIENT_PHONE_NUMBER = os.getenv('RECIPIENT_PHONE_NUMBER')
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# MongoDB Configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
client = MongoClient(MONGO_URI)
db = client.careassist

# Initialize Fall Detector
fall_detector = FallDetector()

# Global variable for video capture
video_capture = None

def get_video_capture():
    global video_capture
    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            raise RuntimeError("Could not open video capture")
    return video_capture

def release_video_capture():
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None

def generate_frames():
    try:
        cap = get_video_capture()
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            try:
                # Process frame for fall detection
                frame, fall_detected = fall_detector.process_frame(frame)
                
                # Convert frame to bytes for streaming
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue
    except Exception as e:
        print(f"Error in generate_frames: {str(e)}")
        release_video_capture()

@app.route('/')
def dashboard():
    reminders = list(db.reminders.find().sort('reminder_time', 1))
    return render_template('dashboard.html', reminders=reminders)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video')
def stop_video():
    release_video_capture()
    return {'status': 'success'}

@app.route('/add_reminder', methods=['POST'])
def add_reminder():
    try:
        reminder = {
            'medication_name': request.form['medication'],
            'dosage': request.form['dosage'],
            'reminder_time': request.form['time'],
            'frequency': request.form['frequency'],
            'phone_number': request.form['phone'].replace(" ", ""),
            'created_at': datetime.now()
        }
        db.reminders.insert_one(reminder)
        return redirect('/')
    except Exception as e:
        return str(e), 500

@app.route('/delete_reminder/<reminder_id>')
def delete_reminder(reminder_id):
    try:
        db.reminders.delete_one({'_id': reminder_id})
        return redirect('/')
    except Exception as e:
        return str(e), 500

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        phone_number = request.form['phone'].replace(" ", "")
        message = request.form['message']
        
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        return redirect('/')
    except Exception as e:
        return str(e), 500

@app.route('/voice_call', methods=['POST'])
def voice_call():
    try:
        phone_number = request.form['phone'].replace(" ", "")
        message = request.form['message']
        
        twilio_client.calls.create(
            to=phone_number,
            from_=TWILIO_PHONE_NUMBER,
            url=url_for('voice_response', message=message, _external=True)
        )
        return redirect('/')
    except Exception as e:
        return str(e), 500

@app.route('/voice')
def voice_response():
    message = request.args.get('message', 'Hello! This is a default message.')
    response = VoiceResponse()
    response.say(message, voice='alice')
    return str(response), {'Content-Type': 'text/xml'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)