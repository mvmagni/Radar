from support.BoundingBox import BoundingBox
from dataclasses import dataclass

# Detected objects in image
@dataclass
class DetectedObject:
    
    # Required fields
    bbox: BoundingBox
    classID: int
    className: str
    confidence: float
    object_type: str
    
    def draw(self):
        pass
    
    
    