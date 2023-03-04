from support.OperatingConfig import OperatingConfig
from support.ModelNet import ModelNet
from support.Hands import Hands
from support.DrawingManager import DrawingManager
from support.DetectedObject import DetectedObject

class FrameManager:
    
    def __init__(self, 
                 operatingConfig: OperatingConfig):     
        self.operatingConfig = operatingConfig
        self.modelNet = self.create_modelNet()
        self.mpHands = self.create_hand_track()
        self.drawingManager = self.create_drawingManager()
        
    def create_drawingManager(self):
        return DrawingManager(operatingConfig=self.operatingConfig,
                              mpHands=self.mpHands)
    
    def create_modelNet(self):
        return ModelNet(config_dir=self.operatingConfig.model_config_dir,
                        classname_file=self.operatingConfig.className_file,
                        model_type=self.operatingConfig.detection_model,
                        confidence_threshold=self.operatingConfig.CONFIDENCE_THRESHOLD,
                        nms_threshold=self.operatingConfig.NMS_THRESHOLD,
                        object_type=self.operatingConfig.OBJECT_GENERIC
                       )
    
    def create_hand_track(self):
        return Hands(object_type=self.operatingConfig.OBJECT_HAND)
    
    def process_image(self, img):
        detected_objects: list[DetectedObject] = []
        
        
        if self.operatingConfig.MODELNET_DETECT:
            # Check to see if current detection model matches the desired model
            if not (self.operatingConfig.detection_model == self.modelNet.model_type):
                self.modelNet = self.create_modelNet()

            # Make sure all variable are up to date before detection
            if not (self.operatingConfig.NMS_THRESHOLD == self.modelNet.nms_threshold):
                self.modelNet.nms_threshold = self.operatingConfig.NMS_THRESHOLD
            
            if not (self.operatingConfig.CONFIDENCE_THRESHOLD == self.modelNet.confidence_threshold):
                self.modelNet.confidence_threshold = self.operatingConfig.CONFIDENCE_THRESHOLD
            
            # Returns a list[DetectedObject]
            detected_objects += self.modelNet.detect(img=img)
            #print(type(modelNet_objects))
            #print(type(detected_objects))
            #detected_objects = detected_objects + modelNet_objects
            
            
        if self.operatingConfig.FIND_HANDS:
            detected_objects += self.mpHands.findHands(img=img)
            
               
        if self.operatingConfig.draw_results():
            img = self.drawingManager.draw_results(img=img, 
                                                   detected_objects=detected_objects)

        return img
    
    def write_config_screen(self,
                            img):
        self.drawingManager.write_config_screen(img=img,
                                                modelNet=self.modelNet)
    
    def write_loading_model(self,
                            img):
        self.drawingManager.write_loading_model(img=img)