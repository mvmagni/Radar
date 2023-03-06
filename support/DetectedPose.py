from support.DetectedObject import DetectedObject
from typing import Any
#from support.Hands import Hands
from dataclasses import dataclass


# Detected objects in image
@dataclass
class DetectedPose(DetectedObject):
    
    # Required fields (in addition to parent class)
    pose_keypoints: Any
    
    def draw(self):
        pass