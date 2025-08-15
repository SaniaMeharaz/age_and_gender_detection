import cv2
import math
import argparse

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
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Set up video capture
video = cv2.VideoCapture(args.image if args.image else 0)

padding = 20

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

        # Age prediction
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # Display result
        label = f"{gender}, {age}"
        print(f"Detected: {label}")
        cv2.putText(resultImg, label, (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Age and Gender Detection", resultImg)

# Release video resources
video.release()
cv2.destroyAllWindows()
