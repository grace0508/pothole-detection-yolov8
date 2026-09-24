from dotenv import load_dotenv
import os
import cv2
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import time

# Load environment variables
load_dotenv("create.env")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# Email sender
def send_email(frame_number, confidence, frame, pothole_id):
    msg = MIMEMultipart()
    msg['Subject'] = "Pothole Detected - Urgent Repair Required"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    body = f"A pothole was detected in frame {frame_number} with confidence {confidence:.2f}.\nPlease investigate."
    msg.attach(MIMEText(body, 'plain'))

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

# Load model
model = YOLO('best.pt')

video_path = "car.webm"
if not os.path.exists(video_path):
    print(f"Video {video_path} not found")
    exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Failed to open video.")
    exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("pothole_detected_output.mp4", fourcc, fps, (frame_width, frame_height))

frame_count = 0
email_cooldown = 60  # Cooldown in seconds
last_email_times = {}  # Track last email time per pothole ID

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Use model.track instead of model.predict for persistent tracking
    results = model.track(frame, imgsz=640, persist=True)

    for result in results:
        for box in result.boxes:
            if box.cls is not None and int(box.cls) == 0 and box.id is not None:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                track_id = int(box.id.item())  # Get the tracking ID
                pothole_id = f"pothole_{track_id}"  # Use tracking ID for consistency

                # Draw the annotation with tracking ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame, f"Pothole ID: {track_id} Conf: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 234), 2)

                # Send email if confidence ≥ 0.5 and respecting cooldown
                if conf >= 0.5:
                    current_time = time.time()
                    if pothole_id not in last_email_times or (current_time - last_email_times[pothole_id] > email_cooldown):
                        send_email(frame_count, conf, frame, pothole_id)
                        last_email_times[pothole_id] = current_time

    out.write(frame)
    cv2.imshow("Pothole Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print(f"Processed frame {frame_count}/{total_frames}")

cap.release()
out.release()
cv2.destroyAllWindows()