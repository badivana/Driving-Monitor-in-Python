#Credits: Prajwal Badivana Git: Badivana

import cv2
from utils import *
from detection.face import *
from detection.pose import *
from state import *
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import urllib.request
import os

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face_landmarker.task model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")

def main():

    """ Main function to monitor the driver's state and detect signs of drowsiness. """

    download_model()

    # Thresholds defined for driver state evaluation
    marThresh = 0.7
    marThresh2 = 0.15
    headThresh = 6
    earThresh = 0.28
    blinkThresh = 10
    gazeThresh = 5

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Build FaceLandmarker (new mediapipe API)
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    faceLandmarker = mp_vision.FaceLandmarker.create_from_options(options)

    captureFps = cap.get(cv2.CAP_PROP_FPS)
    if captureFps == 0:
        captureFps = 30  # fallback for webcams that report 0

    driverState = DriverState(marThresh, marThresh2, headThresh, earThresh, blinkThresh, gazeThresh)
    headPose = HeadPose(faceLandmarker)
    faceDetector = FaceDetector(faceLandmarker, captureFps, marThresh, marThresh2, headThresh, earThresh, blinkThresh)

    startTime = time.time()
    drowsinessCounter = 0
    lastAlertTime = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame, results = headPose.process_image(frame)
        frame = headPose.estimate_pose(frame, results, True)
        roll, pitch, yaw = headPose.calculate_angles()

        frame, sleepEyes, mar, gaze, yawning, baseR, baseP, baseY, baseG = faceDetector.evaluate_face(frame, results, roll, pitch, yaw, True)

        frame, state = driverState.eval_state(frame, sleepEyes, mar, roll, pitch, yaw, gaze, yawning, baseR, baseP, baseG)

        # Update drowsiness counter if the driver is drowsy
        if state == "Drowsy":
            drowsinessCounter += 1
        else:
            if drowsinessCounter > 0:
                drowsinessCounter -= 1

        drowsinessTime = drowsinessCounter / captureFps
        drowsy = drowsinessTime / 60

        # Reset the drowsiness counter after 1 minute
        if time.time() - startTime >= 60:
            drowsinessCounter = 0
            startTime = time.time()

        cv2.imshow('Driver State Monitoring', frame)

        # Alert every 3 seconds while drowsy, stop when alert clears
        if drowsy > 0.08:
            if time.time() - lastAlertTime >= 1:
                print("USER IS SHOWING SIGNALS OF DROWSINESS. SENDING ALERT")
                lastAlertTime = time.time()
        else:
            lastAlertTime = 0

        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()