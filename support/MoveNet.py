import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from support.DetectedPose import DetectedPose
from support.BoundingBox import BoundingBox

class MoveNet:
    
    def __init__(self, object_type):
        self.model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
        self.movenet = self.model.signatures['serving_default']
        self.object_type = object_type
        
        self.edges = {
            (0, 1): 'm',
            (0, 2): 'c',
            (1, 3): 'm',
            (2, 4): 'c',
            (0, 5): 'm',
            (0, 6): 'c',
            (5, 7): 'm',
            (7, 9): 'm',
            (6, 8): 'c',
            (8, 10): 'c',
            (5, 6): 'y',
            (5, 11): 'm',
            (6, 12): 'c',
            (11, 12): 'y',
            (11, 13): 'm',
            (13, 15): 'm',
            (12, 14): 'c',
            (14, 16): 'c'
        }

    def findPoses(self, img)-> list[DetectedPose]:
        # Resize image
        img_tf = img.copy()
        img_tf = tf.image.resize_with_pad(tf.expand_dims(img_tf, axis=0), 384,640)
        input_img = tf.cast(img_tf, dtype=tf.int32)
        
        # Detection section
        results = self.movenet(input_img)
        
        keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))

        detected_poses: list[DetectedPose] = []

        for pose_keypoints in keypoints_with_scores:
            detected_poses.append(self.get_boundingBox(img=img,
                                                       pose_keypoints=pose_keypoints))

        return detected_poses
    
    def draw_poses(self, img, pose_keypoints, confidence_threshold):
        y, x, c = img.shape
        shaped = np.squeeze(np.multiply(pose_keypoints, [y,x,1]))
        
        #Draw keypoints
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(img, (int(kx), int(ky)), 6, (0,255,0), -1)
        
        # Draw connections
        for edge, color in self.edges.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]
            
            if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)
        
    def get_boundingBox(self, img, pose_keypoints) -> DetectedPose:
        y, x, c = img.shape
        shaped = np.squeeze(np.multiply(pose_keypoints, [y,x,1]))
        
        x_min = None
        x_max = None
        y_min = None
        y_max = None
        confidence_min = None
        confidence_max = None
        
        for edge, color in self.edges.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]
            
            if x_min is None or x_max is None or y_min is None or y_max is None:
                x_min = min(x1, x2)
                x_max = max(x1, x2)
                y_min = min(y1, y2)
                y_max = max(y1, y2)
                confidence_min = min(c1, c2)
                confidence_max = max(c1, c2)
            else:
                x_min = min(x_min, x1, x2)
                x_max = max(x_max, x1, x2)
                y_min = min(y_min, y1, y2)
                y_max = max(y_max, y1, y2)
                confidence_min = min(confidence_min, c1, c2)
                confidence_max = max(confidence_max, c1, c2)
            
            
            
        
        
        confidence = f'{confidence_min}->{confidence_max}'
        
        bbox = BoundingBox(top_left_x=int(x_min),
                           top_left_y=int(y_min),
                           width=int(x_max - x_min),
                           height=int(y_max - y_min))
        
        return DetectedPose(bbox = bbox,
                            classID=None,
                            className='pose',
                            confidence=confidence,
                            object_type=self.object_type,
                            pose_keypoints=pose_keypoints)
