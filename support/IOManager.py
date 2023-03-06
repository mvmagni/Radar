import cv2 as cv
#import numpy as np
from datetime import datetime

from PIL import Image

#from support.model_net import ModelNet
import support.yolo_config as yc
from support.OperatingConfig import OperatingConfig

class IOManager:
    
    def __init__(self,operatingConfig: OperatingConfig):
        self.operatingConfig = operatingConfig

    def write_progress_image(self,
                             img, 
                             extension='jpg', 
                             date_format='%Y-%m-%d_%H-%M-%S'):
        
        fileName = f"{datetime.today().strftime(f'{date_format}_f{self.operatingConfig.frame_counter}.{extension}')}"
        writeToFile = f'{self.operatingConfig.image_store_dir}/{fileName}'
        
        cv.imwrite(filename=writeToFile, img=img)
        
        image_show = Image.open(writeToFile)
        image_show.show()
            

            
    def handle_general_key_input(self,
                                 img,
                                 key):
        if key%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            self.operatingConfig.RUN_PROGRAM = False
            self.operatingConfig.PROCESS_IMAGES = False
            self.operatingConfig.CONFIGURE = False
        elif key%256 == 102: #small f
            self.operatingConfig.SHOW_FPS = not self.operatingConfig.SHOW_FPS
        elif key%256 == 100: #small d
            self.operatingConfig.MODELNET_DETECT = not self.operatingConfig.MODELNET_DETECT
        elif key%256 == 98: #small b
            self.operatingConfig.SHOW_BOUNDING_BOXES = not self.operatingConfig.SHOW_BOUNDING_BOXES    
        elif key%256 == 108: # small l (letter L)
            self.operatingConfig.SHOW_LABELS = not self.operatingConfig.SHOW_LABELS
        elif key%256 == 104: # small h 
            self.operatingConfig.SHOW_HANDS = not self.operatingConfig.SHOW_HANDS
        elif key%256 == 72: # capital H
            self.operatingConfig.FIND_HANDS = not self.operatingConfig.FIND_HANDS
        
        elif key%256 == 112: # small p 
            print("show pose toggled")
            self.operatingConfig.SHOW_POSE = not self.operatingConfig.SHOW_POSE
        elif key%256 == 80: # capital P
            print("find pose toggled")
            self.operatingConfig.FIND_POSE = not self.operatingConfig.FIND_POSE
            
        elif key%256 == 105: # small i (letter eye)
            self.operatingConfig.SHOW_RUNTIME_CONFIG = not self.operatingConfig.SHOW_RUNTIME_CONFIG
        
        # Model changes - cycle through available
        elif key%256 == 91: # left square bracket "["
            # Set previous model as desired one
            self.set_desired_model(-1)
            self.operatingConfig.PROCESS_IMAGES = False  
        elif key%256 == 93: # right square bracket "]"
            # Set next model as desired one
            self.set_desired_model(1)
            self.operatingConfig.PROCESS_IMAGES = False
            
        # Increase or decrease confidence threshold
        elif key%256 == 43: # + increase confidence    
            self.adjust_confidence_threshold(adjust_by=self.operatingConfig.CONF_THRESHOLD_ADJUSTBY)
            self.operatingConfig.increment_show_info_counter()
        elif key%256 == 45: # - decrease confidence
            self.adjust_confidence_threshold(adjust_by=-self.operatingConfig.CONF_THRESHOLD_ADJUSTBY)
            self.operatingConfig.increment_show_info_counter()
        # Increase or decrease confidence threshold
        elif key%256 == 39: # ' increase nms_threshold    
            self.adjust_nms_threshold(adjust_by=self.operatingConfig.NMS_THRESHOLD_ADJUSTBY)
            self.operatingConfig.increment_show_info_counter()
        elif key%256 == 59: # ; decrease nms_threshold
            self.adjust_nms_threshold(adjust_by=-self.operatingConfig.NMS_THRESHOLD_ADJUSTBY)
            self.operatingConfig.increment_show_info_counter()
        # Take screenshot
        elif key%256 == 32: # SPACE pressed
            self.write_progress_image(img=img)   
    

    def handle_config_key_input(self,
                                img,
                                key):

            
        if key%256 == 103: # letter g, Go and process video
            print(f'Running process')
            self.operatingConfig.CONFIGURE = False
        
        
        self.handle_general_key_input(img=img,
                                      key=key)
        
    
    def handle_processing_key_input(self,
                                    img,
                                    key):
        if key%256 == 99: #small c
            self.operatingConfig.PROCESS_IMAGES = False
            self.operatingConfig.CONFIGURE = True

        self.handle_general_key_input(img=img,
                                      key=key)



    def adjust_nms_threshold(self,
                             adjust_by):
        
        new_threshold = round(self.operatingConfig.NMS_THRESHOLD + adjust_by,2)
        
        if new_threshold < 0: #Can't go below 0
            # Adding the operating config threshold stays if model changes
            self.operatingConfig.NMS_THRESHOLD = 0
            #self.operatingConfig.modelNet.nms_threshold = 0
        elif new_threshold > 1: #Can't go above 1
            # Adding the operating config threshold stays if model changes
            self.operatingConfig.NMS_THRESHOLD = 1
            #self.operatingConfig.modelNet.nms_threshold = 1
        else:
            # Adding the operating config threshold stays if model changes
            self.operatingConfig.NMS_THRESHOLD = new_threshold
            #self.operatingConfig.modelNet.nms_threshold = new_threshold


    def adjust_confidence_threshold(self,
                                    adjust_by):
        
        new_threshold = round(self.operatingConfig.CONFIDENCE_THRESHOLD + adjust_by,2)
        
        if new_threshold < 0: #Can't go below 0
            # Adding the operating config threshold stays if model changes
            self.operatingConfig.CONFIDENCE_THRESHOLD = 0
            #self.operatingConfig.modelNet.confidence_threshold = 0
        elif new_threshold > 1: #Can't go above 1
            # Adding the operating config threshold stays if model changes
            self.operatingConfig.CONFIDENCE_THRESHOLD = 1
            #self.operatingConfig.modelNet.confidence_threshold = 1
        else:
            # Adding the operating config threshold stays if model changes
            self.operatingConfig.CONFIDENCE_THRESHOLD = new_threshold
            #self.operatingConfig.modelNet.confidence_threshold = new_threshold


    def set_desired_model(self,
                          increment):
        
        desired_model=self.operatingConfig.detection_model
        model_list = yc.MODEL_LIST
        
        curr_model_index=model_list.index(desired_model)    
        
        # deal with edge cases first
        if curr_model_index == (len(model_list)-1) and increment == 1: # Case end of list
            print(f'Current index at end of list')
            desired_model=model_list[0]
        elif curr_model_index == 0 and increment == -1: # Case start of list
            print(f'Current index at start of list')
            desired_model=model_list[len(model_list)-1]
        else:
            print(f'Standard case: increment by: {increment}')
            desired_model=model_list[curr_model_index+increment]
        
        self.operatingConfig.detection_model=desired_model
        print(f'Desired model: {desired_model}')




