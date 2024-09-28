import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import handle_image
import time

VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

current_frame = None

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global current_frame
    annotated_image = handle_image.draw_landmarks_on_image(output_image.numpy_view(), result)
    current_frame = cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR)


base_options = python.BaseOptions(model_asset_path=r'C:\Users\jakub\Downloads\hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1,
                                       running_mode=VisionRunningMode.LIVE_STREAM,
                                       result_callback=print_result)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv.VideoCapture(0)

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    timestamp = int(time.time() * 1000)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    detector.detect_async(mp_image, timestamp)
    if current_frame is not None:
        cv.imshow('Hand Tracking', current_frame)
    else:
        cv.imshow('Hand Tracking', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()