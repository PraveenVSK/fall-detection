import cv2
import numpy as np
import time
from inference_sdk import InferenceHTTPClient
from twilio.rest import Client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv('ROBOFLOW_API_KEY')
)

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
RECIPIENT_PHONE_NUMBER = os.getenv('RECIPIENT_PHONE_NUMBER')

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

COOLDOWN_TIME = 60  # 1 minute cooldown between alerts
CONFIDENCE_THRESHOLD = 0.5  # Adjusted threshold for better accuracy

class FallDetector:
    def __init__(self):
        self.last_alert_time = 0
        self.fall_buffer = []
        print("FallDetector initialized with confidence threshold:", CONFIDENCE_THRESHOLD)

    def check_cooldown(self):
        return (time.time() - self.last_alert_time) < COOLDOWN_TIME

    def update_buffer(self, detected):
        self.fall_buffer.append(detected)
        if len(self.fall_buffer) > 5:  # Reduced buffer size for quicker response
            self.fall_buffer.pop(0)
        print(f"Fall buffer updated: {sum(self.fall_buffer)}/{len(self.fall_buffer)} detections")

    def should_alert(self):
        if len(self.fall_buffer) < 3:  # Require fewer detections to alert
            return False
        alert_ratio = sum(self.fall_buffer) / len(self.fall_buffer)
        print(f"Alert ratio: {alert_ratio}")
        return alert_ratio > 0.5 and not self.check_cooldown()  # Lower ratio required

    def process_frame(self, frame):
        fall_detected = False
        try:
            # Resize frame for better performance
            frame = cv2.resize(frame, (640, 640))
            
            # Convert frame to bytes for Roboflow API
            _, img_encoded = cv2.imencode('.jpg', frame)
            
            # Make API call
            result = CLIENT.infer(img_encoded.tobytes(), model_id="fall-detection-ca3o8/4")
            print("Roboflow API Response:", result)
            
            # Process predictions
            predictions = result.get("predictions", [])
            
            for pred in predictions:
                confidence = pred.get("confidence", 0)
                class_name = pred.get("class", "unknown").lower()  # Make case-insensitive
                print(f"Prediction: {class_name} with confidence {confidence}")
                
                # Check for variations of "fall" in class name
                if "fall" in class_name and confidence > CONFIDENCE_THRESHOLD:
                    fall_detected = True
                    
                    # Handle different bounding box structures
                    box = pred.get("bbox", {}) or pred.get("bear", {})
                    x = int(box.get("x", 0))
                    y = int(box.get("y", 0))
                    width = int(box.get("width", 0))
                    height = int(box.get("height", 0))
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
                    cv2.putText(frame, f"Fall: {confidence:.2f}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Update fall detection buffer
            self.update_buffer(fall_detected)
            
            # Check if should send alert
            if self.should_alert():
                try:
                    print("Sending Twilio alert...")
                    twilio_client.messages.create(
                        body="ALERT: Fall detected! Immediate assistance required!",
                        from_=TWILIO_PHONE_NUMBER,
                        to=RECIPIENT_PHONE_NUMBER
                    )
                    print("Twilio alert sent successfully")
                    self.last_alert_time = time.time()
                    self.fall_buffer = []
                except Exception as e:
                    print(f"Error sending SMS: {str(e)}")

            # Add status text to frame
            status_text = "FALL DETECTED!" if fall_detected else "Monitoring..."
            color = (0, 0, 255) if fall_detected else (0, 255, 0)
            cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Confidence Threshold: {CONFIDENCE_THRESHOLD}", (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            return frame, fall_detected

        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame, False