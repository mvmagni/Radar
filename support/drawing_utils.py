import cv2 as cv
from support.BoundingBox import BoundingBox

def show_bounding_box_modelNet(img, 
                               bbox: BoundingBox, 
                               classID, 
                               class_name, 
                               confidence, 
                               show_labels, 
                               weight=1):
    
    
    draw_bounding_box(img=img,
                      bbox=bbox,
                      classID=classID,
                      weight=weight)
    
    if show_labels:
        x, y = bbox.get_top_left()
        
        show_label(img=img,
                   x=x,
                   y=y,
                   class_name=class_name,
                   confidence=confidence,
                   classID=classID)    
    
    return img

def draw_bounding_box(img,
                      bbox: BoundingBox, 
                      classID, 
                      weight=1):
    
    colour=get_class_colour(classID)
    
    x,y = bbox.get_top_left()
    w = bbox.width
    h = bbox.height
    
    cv.rectangle(img,(x,y), (x+w, y+h), colour, weight)
    
    line_width_max = 50
    line_width = min(int(bbox.width/2 * 0.30), line_width_max)
    line_height = min(int(bbox.height/2 * 0.30), line_width_max)
    line_thickness_w = 3
    line_thickness_h = 3
    
    # Top left
    cv.line(img, (x,y), (x + line_width, y), colour, thickness=line_thickness_w)
    cv.line(img, (x,y), (x, y + line_height), colour, thickness=line_thickness_h)
    
    # Top right
    cv.line(img, (x + w,y), (x + w - line_width, y), colour, thickness=line_thickness_w)
    cv.line(img, (x + w,y), (x + w, y + line_height), colour, thickness=line_thickness_h)
    
    # Bottom left
    cv.line(img, (x,y + h), (x + line_width, y + h), colour, thickness=line_thickness_w)
    cv.line(img, (x,y + h), (x, y + h - line_height), colour, thickness=line_thickness_h)
    
    # Bottom right
    cv.line(img, (x + w, y + h), (x + w - line_width, y + h), colour, thickness=line_thickness_w)
    cv.line(img, (x + w, y + h), (x + w, y + h - line_height), colour, thickness=line_thickness_h)
    
    
    
    
    return img


def show_label(img, 
               x,
               y,
               class_name,
               confidence: int,
               classID):
    
    if confidence is None:
        confidence_label="NA"
    else:
        confidence_label=f'{class_name} {int(confidence*100)}%'
    
    colour = get_class_colour(classID=classID)
    cv.putText(img,
               confidence_label,
               (x,y-5), 
               cv.FONT_HERSHEY_SIMPLEX,
               0.6,
               colour,
               2
               )
    return img

def get_class_colour(classID):
    if classID == 0: # person
        colour = (255,51,153) # colour=blue
    elif classID == 2: # car
        colour = (255,153,51) # colour=cyan
    else: 
        colour = (0,204,204) # colour=yellow
    
    return colour

