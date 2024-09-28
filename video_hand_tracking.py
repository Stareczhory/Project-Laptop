import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import handle_image

VisionRunningMode = mp.tasks.vision.RunningMode


base_options = python.BaseOptions(model_asset_path=r'C:\Users\jakub\Downloads\hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1,
                                       running_mode=VisionRunningMode.VIDEO)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv.VideoCapture(r'C:\Users\jakub\Downloads\signal-2024-09-27-192348.mp4')

frame_rate = cap.get(cv.CAP_PROP_FPS)
time_per_frame_ms = (1 / frame_rate)*1000
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    timestamp = frame_number * time_per_frame_ms
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    detection_result = detector.detect_for_video(mp_image, int(timestamp))
    annotated_image = handle_image.draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    cv.imshow('img', cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))


    if cv.waitKey(1) == ord('q'):
        break

    frame_number += 1

cap.release()
cv.destroyAllWindows()
