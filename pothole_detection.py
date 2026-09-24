from dotenv import load_dotenv
import os
import cv2
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import time
import numpy as np

# Load environment variables
load_dotenv("create.env")
print("EMAIL_SENDER:", os.getenv("EMAIL_SENDER"))
print("EMAIL_PASSWORD:", os.getenv("EMAIL_PASSWORD"))
print("EMAIL_RECEIVER:", os.getenv("EMAIL_RECEIVER"))
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# Function to send email notification with image
def send_email(frame_number, confidence, frame, pothole_id):
    msg = MIMEMultipart()
    msg['Subject'] = "Pothole Detected - Urgent Repair Required"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    # Email body
    body = f"A pothole was detected in frame {frame_number} with confidence {confidence:.2f}.\nPlease investigate and repair at the earliest."
    msg.attach(MIMEText(body, 'plain'))

    # Attach image
    _, img_encoded = cv2.imencode('.jpg', frame)
    img = MIMEImage(img_encoded.tobytes())
    img.add_header('Content-Disposition', 'attachment', filename=f'pothole_frame_{frame_number}.jpg')
    msg.attach(img)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print(f"Email sent for pothole {pothole_id} in frame {frame_number}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Load pre-trained model
model = YOLO('best.pt')  # Replace with path to your custom model if different

# Process video
video_path = "car.webm"  # Adjust path as needed
print(f"Looking for video at: {video_path}")
if not os.path.exists(video_path):
    print(f"Video file {video_path} not found")
    exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Failed to open video {video_path}")
    exit(1)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = "pothole_detected_output.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
email_cooldown = 60  # Seconds between emails for the same pothole
pothole_times = {}  # Dictionary to track last email time per pothole (identified by coordinates and confidence)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Run inference
    results = model(frame, imgsz=640)

    # Process detections
    for result in results:
        for box in result.boxes:
            if box.cls is not None and int(box.cls) == 0:  # Pothole class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                # Unique pothole ID based on coordinates and confidence
                pothole_id = f"{x1}_{y1}_{x2}_{y2}_{conf:.2f}"

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 8)
                cv2.putText(frame, f"Pothole: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check cooldown for this specific pothole
                current_time = time.time()
                if pothole_id not in pothole_times or (current_time - pothole_times[pothole_id] >= email_cooldown):
                    send_email(frame_count, conf, frame, pothole_id)
                    pothole_times[pothole_id] = current_time

    # Display frame in real-time
    cv2.imshow('Pothole Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

    # Write frame to output video
    out.write(frame)

    # Print progress
    print(f"Processed frame {frame_count}/{total_frames}")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to {output_path}")