from support.DetectedObject import DetectedObject
#from support.Hands import Hands
from dataclasses import dataclass


# Detected objects in image
@dataclass
class DetectedHand(DetectedObject):
    
    # Required fields (in addition to parent class)
    hand_landmarks: list
    #mpHand: Hands
    
    
    def draw(self):
        pass