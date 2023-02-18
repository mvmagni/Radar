from support.BoundingBox import BoundingBox
from dataclasses import dataclass, field

@dataclass
class Object:
    
    # Required fields
    bbox: BoundingBox
    
    
    # Optional fields
    