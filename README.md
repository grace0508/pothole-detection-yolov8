
## 📌 Project Overview

Potholes can cause road accidents, vehicle damage, and traffic problems. Manual inspection of roads can be time-consuming and difficult to perform continuously.

This project uses **YOLOv8 object detection** to automatically identify potholes from road videos.

When a pothole is detected:

1. The pothole is identified using the trained YOLOv8 model.
2. A bounding box is drawn around the detected pothole.
3. The confidence score is displayed.
4. The detection frame is captured.
5. An email notification is sent to the configured recipient.
6. The processed video is saved as an output file.

The project combines **computer vision, deep learning, video processing, and automated email notification** into a single application.

---

## 🎯 Objectives

- To automatically detect potholes using deep learning.
- To process road videos frame by frame.
- To identify potholes using a trained YOLOv8 model.
- To display bounding boxes around detected potholes.
- To display the confidence score of each detection.
- To generate a processed output video.
- To automatically send email alerts when potholes are detected.
- To attach the detected pothole image to the notification email.
- To reduce the need for continuous manual road inspection.

---

## ✨ Features

### 🔍 Pothole Detection

The system uses a trained **YOLOv8 object detection model** to identify potholes in road images and video frames.

### 🎥 Video Processing

The input road video is processed frame by frame using OpenCV.

### 📦 Bounding Box Detection

Detected potholes are marked with bounding boxes and their confidence scores are displayed.

### 📊 Confidence Score

For every detected pothole, the system obtains and displays the model's confidence score.

### 📧 Email Notification

When a pothole is detected, the system automatically sends an email notification through Gmail SMTP.

The email contains:

- Detection frame number
- Detection confidence
- Description of the detection
- Image of the detected pothole

### ⏱️ Email Cooldown

The system uses a cooldown mechanism to reduce repeated email notifications for the same detected pothole.

The current cooldown period is:

```text
60 seconds
💾 Output Video
The processed video is saved as:
pothole_detected_output.mp4
________________________________________
🛠️ Technologies Used
Technology	Purpose
Python	Main programming language
YOLOv8	Pothole object detection
Ultralytics	YOLOv8 implementation
OpenCV	Image and video processing
NumPy	Numerical operations
Python-dotenv	Environment variable management
SMTP	Email notification
Gmail SMTP	Sending email alerts
VS Code	Development environment
Git & GitHub	Version control and project hosting
________________________________________
🏗️ System Workflow
              Input Road Video
                     │
                     ▼
              Read Video Frame
                     │
                     ▼
              YOLOv8 Detection
                     │
                     ▼
            Is Pothole Detected?
                /          \
              No            Yes
              │              │
              │              ▼
              │       Draw Bounding Box
              │              │
              │              ▼
              │      Calculate Confidence
              │              │
              │              ▼
              │       Capture Detection
              │              │
              │              ▼
              │        Send Email Alert
              │              │
              │              ▼
              │       Save Processed Frame
              │              │
              └──────────────┘
                     │
                     ▼
                Output Video
________________________________________
📂 Project Structure
pothole-detection-yolov8/
│
├── detect.py
├── best.pt
├── car.webm
├── requirements.txt
├── README.md
├── .gitignore
└── create.env
create.env contains email credentials and should not be uploaded to a public repository.
________________________________________
🧠 YOLOv8 Model
The project uses a trained YOLOv8 model:
model = YOLO('best.pt')
The best.pt file contains the trained model weights used for pothole detection.
The model analyzes each video frame and provides:
•	Detected objects
•	Bounding box coordinates
•	Class information
•	Confidence scores
________________________________________
🎥 Input Video
The system processes a road video specified in the Python program:
video_path = "car.webm"
The program checks whether the input video exists before processing it.
________________________________________
🔍 Detection Process
For every frame, YOLOv8 performs inference:
results = model(frame, imgsz=640)
The model processes the frame using an image size of 640.
The pothole class is identified using the class ID configured in the trained model.
The bounding box coordinates are obtained from the YOLO detection results, and the confidence score is calculated for each detection.
________________________________________
📦 Bounding Box and Label
When a pothole is detected, OpenCV draws a bounding box around it and displays the confidence score.
Example:
Pothole: 0.87
The value represents the confidence score returned by the YOLOv8 model.
________________________________________
📧 Email Notification System
The system uses Gmail SMTP to automatically send notifications when potholes are detected.
SMTP server:
smtp.gmail.com
SMTP port:
465
The email contains:
•	Detection frame number
•	Confidence score
•	Detection description
•	Captured pothole image
The detected frame is converted into JPEG format and attached to the email.
________________________________________
🔐 Environment Variables
Email credentials are loaded from the environment file using python-dotenv.
The application uses:
EMAIL_SENDER
EMAIL_PASSWORD
EMAIL_RECEIVER
Example local create.env file:
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_email_app_password
EMAIL_RECEIVER=receiver_email@gmail.com
Never upload create.env to GitHub.
Add the following to .gitignore:
create.env
.env
Security Note
Do not print email credentials in the Python program.
Avoid code such as:
print("EMAIL_PASSWORD:", os.getenv("EMAIL_PASSWORD"))
because it exposes the password in the terminal.
________________________________________
⏱️ Email Cooldown Mechanism
The system uses a cooldown period to prevent repeated email notifications.
email_cooldown = 60
The program maintains a dictionary of previously detected potholes and their notification times.
This helps reduce excessive email notifications when the same pothole remains visible across multiple video frames.
________________________________________
💾 Output Video
The processed video is saved as:
pothole_detected_output.mp4
The output contains:
•	Original video frames
•	Detected pothole bounding boxes
•	Confidence scores
________________________________________
⚙️ Requirements
The main Python libraries required are:
ultralytics
opencv-python
python-dotenv
numpy
Python's built-in modules used include:
os
time
smtplib
email
________________________________________
📄 requirements.txt
Create a file named requirements.txt containing:
ultralytics
opencv-python
python-dotenv
numpy
Install the dependencies using:
pip install -r requirements.txt
________________________________________
🚀 How to Run the Project
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/pothole-detection-yolov8.git
Replace YOUR_USERNAME with your GitHub username.
2. Open the Project
Open the project folder in VS Code.
3. Create a Virtual Environment
On Windows:
python -m venv evm
Activate it:
evm\Scripts\activate
4. Install Dependencies
pip install -r requirements.txt
5. Configure Email
Create a local file named:
create.env
Add:
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECEIVER=receiver_email@gmail.com
Keep this file private.
6. Add the YOLO Model
Place the trained model in the project directory:
best.pt
The program loads it using:
model = YOLO('best.pt')
7. Add the Input Video
Place the input road video in the project directory:
car.webm
8. Run the Program
python detect.py
The program will:
1.	Load the YOLOv8 model.
2.	Open the input video.
3.	Process each frame.
4.	Detect potholes.
5.	Draw bounding boxes.
6.	Display confidence scores.
7.	Send email notifications when required.
8.	Save the processed video.
________________________________________
⌨️ Controls
During real-time video processing:
Press Q
to stop the detection process.
________________________________________
📊 Output
Real-Time Detection
The application displays detected potholes in the video with bounding boxes and confidence scores.
Example:
Pothole: 0.87
Email Alert
When a pothole is detected, an email is sent to the configured recipient with detection information and an image of the detected frame.
Processed Video
The processed video is saved as:
pothole_detected_output.mp4
________________________________________
🧪 Testing
The system can be tested using road videos containing potholes.
Testing includes:
•	Pothole detection
•	Bounding box generation
•	Confidence score display
•	Video processing
•	Email notification
•	Email cooldown
•	Missing input video handling
•	Invalid video handling
•	Model loading
________________________________________
📚 Concepts Learned
This project provided practical experience in:
•	Computer Vision
•	Deep Learning
•	Object Detection
•	YOLOv8
•	OpenCV
•	Video Processing
•	Image Processing
•	Model Inference
•	Confidence Scores
•	Python Programming
•	Environment Variables
•	SMTP Email Communication
•	Automated Notifications
•	Virtual Environments
•	Git and GitHub
________________________________________
🔮 Future Enhancements
Possible future improvements include:
•	Web-based monitoring dashboard
•	GPS location for detected potholes
•	Map-based pothole visualization
•	Mobile notifications
•	Database integration
•	Improved model accuracy using a larger dataset
•	Live camera/video-stream detection
•	Automated pothole detection reports
•	Cloud deployment
•	User authentication
•	Image-based pothole detection
________________________________________
👩‍💻 Author
Nitsa Salu
________________________________________
🎓 Internship Project
This project was developed as part of an AI & Machine Learning Internship at Techgentsia Software Technologies Pvt. Ltd.
The project provided practical experience in applying deep learning and computer vision techniques to a real-world problem: automated pothole detection and notification.
________________________________________
📄 License
This project was developed for educational and internship purposes.

