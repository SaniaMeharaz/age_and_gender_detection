import cv2
import numpy as np
import argparse
from collections import deque

def highlightface(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, scalefactor=1.0, size=(300, 300), mean=[104, 117, 123], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

# Softmax function for better prediction scaling
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Helper to find the best age group based on confidence
def get_age_group(agePreds):
    ageConfidence = softmax(agePreds[0])  # Softmax for better probability scaling
    maxConfidence = np.max(ageConfidence)
    if maxConfidence > 0.6:  # Confidence threshold to avoid unstable predictions
        ageIndex = np.argmax(ageConfidence)
    else:
        # If confidence is lower, choose a weighted average or nearest group
        ageIndex = np.random.choice(len(ageConfidence), p=ageConfidence)
    return ageIndex

# Function to average age predictions for stability
def rolling_average(ages, window_size=5):
    # Convert deque to list before taking the mean
    if len(ages) < window_size:
        return ages[-1]  # Not enough data yet, return the latest age index
    
    averaged_index = int(np.mean(list(ages)[-window_size:]))  # Convert deque to list
    return min(max(averaged_index, 0), len(ageList) - 1)  # Ensure index is within bounds

# Setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--image', help="Path to image file. Leave blank to use webcam.")
args = parser.parse_args()

# Load face, age, and gender models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(17-25)', '(25-32)', '(38-48)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Set up video capture
video = cv2.VideoCapture(args.image if args.image else 0)

padding = 20
age_history = deque(maxlen=5)  # Store last 5 age predictions for rolling average

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    # Detect face in frame
    resultImg, faceBoxes = highlightface(faceNet, frame)
    if not faceBoxes:
        print("No face detected")
        continue

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1]-padding): min(faceBox[3]+padding, frame.shape[0]-1),
                     max(0, faceBox[0]-padding): min(faceBox[2]+padding, frame.shape[1]-1)]

        # Preprocess face for gender and age detection
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Gender prediction
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Age prediction with probabilistic scaling
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        ageIndex = get_age_group(agePreds)  # Using confidence-based selection
        age_history.append(ageIndex)

        # Apply rolling average and ensure it points to a valid age group
        averaged_age_index = rolling_average(age_history)
        stabilized_age = ageList[averaged_age_index]

        # Display result
        label = f"{gender}, {stabilized_age}"
        print(f"Detected: {label}")
        cv2.putText(resultImg, label, (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Age and Gender Detection", resultImg)

# Release video resources
video.release()
cv2.destroyAllWindows()
