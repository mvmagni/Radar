import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import os

print(f'CWD: {os.getcwd()}')

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose

# Setup the Pose function for images - independently for the images standalone processing.
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Setup the Pose function for videos - for video processing.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

# Initialize mediapipe drawing class - to draw the landmarks points.
mp_drawing = mp.solutions.drawing_utils 


def detectPose(image_pose, pose, draw=False, display=False):
    
    original_image = image_pose.copy()
    
    image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)
    
    resultant = pose.process(image_in_RGB)
    
    if resultant.pose_landmarks and draw:
        print(type(resultant))
        print(type(resultant.pose_landmarks))
        #print(resultant.pose_landmarks)
    
        mp_drawing.draw_landmarks(image=original_image, landmark_list=resultant.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
                                                                               thickness=3, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),
                                                                               thickness=2, circle_radius=2))

    return original_image, resultant
    
    # Here we will read our image from the specified path to detect the pose
#image_path = 'Radar/resources/bruce.jpg'
image_path = 'Radar/resources/yoga_pose.jpg'
test_post_image = cv2.imread(image_path)
retImage, resultMarks = detectPose(image_pose=test_post_image, pose=pose_image, draw=True, display=True)
if (retImage is None):
    print("No image returned")
else:
    print(f'image type: {type(retImage)}')
    print(f'resultant: {type(resultMarks.pose_landmarks)}')
cv2.imshow("pose test", retImage)
cv2.waitKey(0)
cv2.destroyAllWindows()