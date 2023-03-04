import cv2 as cv
from dataclasses import dataclass, field
from support.BoundingBox import BoundingBox
import support.yolo_config as yc
from support.DetectedObject import DetectedObject

@dataclass
class ModelNet:
    config_dir: str
    classname_file: str
    model_type: str
    confidence_threshold: float
    nms_threshold: float
    object_type: str
    
    target_whT: int = field(init=False)
    target_hhT: int = field(init=False)
    model_config_file: str = field(init=False)
    model_weights_file: str = field(init=False)
    

    def __post_init__(self):
        self.target_whT, self.target_hhT,self.model_config_file, self.model_weights_file = yc.get_model_config(config_dir=self.config_dir,
                                                                             model_type=self.model_type)
        
        net = cv.dnn.readNet(self.model_weights_file,
                             self.model_config_file)
        
        # Enable GPU CUDA
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        self.model = cv.dnn_DetectionModel(net)

        self.classes = get_classNames(self.classname_file)
        self.model.setInputParams(size=(self.target_whT, self.target_hhT), scale=1/255)
        print(f'ModelNet configuration completed')

    def detect(self, img) -> list:
        frame_objects: list[DetectedObject] = []
        # return example(class_ids, scores, boxes) = detect(img)
        
        (class_ids, scores, boxes) = self.model.detect(img, 
                                                       nmsThreshold=self.nms_threshold,
                                                       confThreshold=self.confidence_threshold)
    
        for idx, box in enumerate(boxes, start=0):
            className = self.classes[class_ids[idx]]
            confidence = scores[idx]
            object_box = BoundingBox(top_left_x=box[0],
                                     top_left_y=box[1],
                                     width=box[2],
                                     height=box[3])
            frame_objects.append(DetectedObject(bbox=object_box,
                                                classID=class_ids[idx],
                                                className=className,
                                                confidence=confidence,
                                                object_type=self.object_type))
        
        return frame_objects
            
    
   
def get_classNames(classFile):
    # Coco info
    classNames = []

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    return classNames