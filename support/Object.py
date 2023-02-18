from support.BoundingBox import BoundingBox
from dataclasses import dataclass, field

# Detected objects in image
@dataclass
class Object:
    
    # Required fields
    bbox: BoundingBox
    
    
    # Optional fields
    