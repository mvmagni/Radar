from support.OperatingConfig import OperatingConfig
from support.BoundingBox import BoundingBox
from support.model_net import ModelNet
from support.Hands import Hands
from support.DrawingManager import DrawingManager


class FrameManager:
    
    def __init__(self, 
                 operatingConfig: OperatingConfig):     
        self.operatingConfig = operatingConfig
        self.modelNet = self.create_modelNet()
        self.drawingManager = self.create_drawingManager()
        self.mpHands = self.create_hand_track()
        
    def create_drawingManager(self):
        return DrawingManager(operatingConfig=self.operatingConfig)
    
    def create_modelNet(self):
        return ModelNet(config_dir=self.operatingConfig.model_config_dir,
                        classname_file=self.operatingConfig.className_file,
                        model_type=self.operatingConfig.detection_model,
                        confidence_threshold=self.operatingConfig.CONFIDENCE_THRESHOLD,
                        nms_threshold=self.operatingConfig.NMS_THRESHOLD
                       )
    
    def create_hand_track(self):
        return Hands(drawingManager=self.drawingManager)
    
    def process_image(self, img):
        

        
        
        if self.operatingConfig.MODELNET_DETECT:
            # Check to see if current detection model matches the desired model
            if not (self.operatingConfig.detection_model == self.modelNet.model_type):
                self.modelNet = self.create_modelNet()

            # Make sure all variable are up to date before detection
            if not (self.operatingConfig.NMS_THRESHOLD == self.modelNet.nms_threshold):
                self.modelNet.nms_threshold = self.operatingConfig.NMS_THRESHOLD
            
            if not (self.operatingConfig.CONFIDENCE_THRESHOLD == self.modelNet.confidence_threshold):
                self.modelNet.confidence_threshold = self.operatingConfig.CONFIDENCE_THRESHOLD
            
            (class_ids, scores, boxes) = self.modelNet.detect(img)
            
            #print(indices)
            for idx, box in enumerate(boxes, start=0):
                className = self.modelNet.classes[class_ids[idx]]
                confidence = scores[idx]
                object_box = BoundingBox(top_left_x=box[0],
                                        top_left_y=box[1],
                                        width=box[2],
                                        height=box[3])
                
                
                img = self.drawingManager.show_bounding_box_modelNet(img=img, 
                                                                     bbox=object_box,
                                                                     classID=class_ids[idx],
                                                                     class_name=f'{className.title()}',
                                                                     confidence=confidence,
                                                                     show_labels=self.operatingConfig.SHOW_LABELS)
        
        if self.operatingConfig.SHOW_HANDS:
            _, img = self.mpHands.findHands(img=img)
        
        
        ###########################################################
        # General visual display info
        if self.operatingConfig.SHOW_FPS:
            self.drawingManager.show_fps(img=img)
        
        if (self.operatingConfig.SHOW_RUNTIME_CONFIG or 
            self.operatingConfig.show_runtime_config_until_frame > self.operatingConfig.frame_counter):

            img = self.drawingManager.write_config_info(img=img,
                                                        modelNet=self.modelNet)
        #=============================================================
        
        return img
    
    def write_config_screen(self,
                            img):
        self.drawingManager.write_config_screen(img=img,
                                                modelNet=self.modelNet)
    
    def write_loading_model(self,
                            img):
        self.drawingManager.write_loading_model(img=img)