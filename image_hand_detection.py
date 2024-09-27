import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import handle_image


# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path=r'C:\Users\jakub\Downloads\hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)


# STEP 3: Load the input image.
image = mp.Image.create_from_file(r'C:\Users\jakub\Downloads\signal-2024-09-27-171425_002.jpeg')

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = handle_image.draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow('img', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

cv2.waitKey(0)

cv2.destroyAllWindows()
