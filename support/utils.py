import cv2 as cv
#import numpy as np
from datetime import datetime
import time
from PIL import Image
from statistics import mean
#from support.model_net import ModelNet
import support.yolo_config as yc
import support.drawing_utils as du
from support.BoundingBox import BoundingBox

def write_progress_image(img, 
                         operating_config, 
                         extension='jpg', 
                         date_format='%Y-%m-%d_%H-%M-%S'):
    
    fileName = f"{datetime.today().strftime(f'{date_format}_f{operating_config.frame_counter}.{extension}')}"
    writeToFile = f'{operating_config.image_store_dir}/{fileName}'
    
    cv.imwrite(filename=writeToFile, img=img)
    
    image_show = Image.open(writeToFile)
    image_show.show()
    
def show_fps(img, operating_config):
    queue_length=60
    
    if operating_config.fps_queue is None:
        operating_config.fps_queue = []
    
    # font which we will be using to display FPS
    font = cv.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()
 
    # Calculating the fps
 
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time - operating_config.prev_frame_time)
    operating_config.prev_frame_time = new_frame_time
 
    # converting the fps into integer
    fps = int(fps)
 
    # add fps
    operating_config.fps_queue.append(fps)
    
    # Remove anything above desired queue/avg length
    while len(operating_config.fps_queue) > queue_length:
        operating_config.fps_queue.pop(0)
 
    # putting the FPS count on the frame
    cv.putText(img, 
               'FPS:'+ str(int(mean((operating_config.fps_queue)))), 
               (15, 50), 
               font, 
               1.0, 
               (100, 255, 0), 
               2, 
               cv.LINE_AA)
        # putting the FPS count on the frame
    cv.putText(img, 
               f'Frame: {str(operating_config.frame_counter)}', 
               (15, 85), 
               font, 
               1.0, 
               (100, 255, 0), 
               2, 
               cv.LINE_AA)
    
def write_info_bottom_left(img,
                           info,
                           operating_config):
    h, w, c = img.shape
    # print(f'w:{w}, h:{h}')
    cv.putText(img, 
               info, 
               (40, h-50), 
               operating_config.font, 
               1.5, 
               (100, 255, 0), 
               2, 
               cv.LINE_AA)

def process_image(img, operating_config):
   
    if operating_config.MODELNET_DETECT:
        (class_ids, scores, boxes) = operating_config.modelNet.detect(img)
        
        #print(indices)
        for idx, box in enumerate(boxes, start=0):
            className = operating_config.modelNet.classes[class_ids[idx]]
            confidence = scores[idx]
            object_box = BoundingBox(top_left_x=box[0],
                                     top_left_y=box[1],
                                     width=box[2],
                                     height=box[3])
            
            
            img = du.show_bounding_box_modelNet(img=img, 
                                                bbox=object_box,
                                                classID=class_ids[idx],
                                                class_name=f'{className.title()}',
                                                confidence=confidence,
                                                show_labels=operating_config.SHOW_LABELS)
    
    if operating_config.SHOW_HANDS:
        _, img = operating_config.mpHands.findHands(img=img)
    
    
    ###########################################################
    # General visual display info
    if operating_config.SHOW_FPS:
        show_fps(img=img, operating_config=operating_config)
    
    if (operating_config.SHOW_RUNTIME_CONFIG or 
        operating_config.show_runtime_config_until_frame > operating_config.frame_counter):

        img = write_config_info(img=img,
                                operating_config=operating_config)
    #=============================================================
    
    
    return img
            



def handle_general_key_input(img,
                             key,
                             operating_config):
    if key%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        operating_config.RUN_PROGRAM = False
        operating_config.PROCESS_IMAGES = False
        operating_config.CONFIGURE = False
    elif key%256 == 102: #small f
        operating_config.SHOW_FPS = not operating_config.SHOW_FPS
    elif key%256 == 100: #small d
        operating_config.MODELNET_DETECT = not operating_config.MODELNET_DETECT
    elif key%256 == 108: # small l (letter L)
        operating_config.SHOW_LABELS = not operating_config.SHOW_LABELS
    elif key%256 == 104:
        operating_config.SHOW_HANDS = not operating_config.SHOW_HANDS
    elif key%256 == 105: # small i (letter eye)
        operating_config.SHOW_RUNTIME_CONFIG = not operating_config.SHOW_RUNTIME_CONFIG
    
    # Model changes - cycle through available
    elif key%256 == 91: # left square bracket "["
        # Set previous model as desired one
        set_desired_model(operating_config, -1)
        operating_config.PROCESS_IMAGES = False  
    elif key%256 == 93: # right square bracket "]"
        # Set next model as desired one
        set_desired_model(operating_config, 1)
        operating_config.PROCESS_IMAGES = False
        
    # Increase or decrease confidence threshold
    elif key%256 == 43: # + increase confidence    
        adjust_confidence_threshold(operating_config=operating_config,
                                    adjust_by=operating_config.CONF_THRESHOLD_ADJUSTBY)
        operating_config.increment_show_info_counter()
    elif key%256 == 45: # - decrease confidence
        adjust_confidence_threshold(operating_config=operating_config,
                                    adjust_by=-operating_config.CONF_THRESHOLD_ADJUSTBY)
        operating_config.increment_show_info_counter()
    # Increase or decrease confidence threshold
    elif key%256 == 39: # ' increase nms_threshold    
        adjust_nms_threshold(operating_config=operating_config,
                             adjust_by=operating_config.NMS_THRESHOLD_ADJUSTBY)
        operating_config.increment_show_info_counter()
    elif key%256 == 59: # ; decrease nms_threshold
        adjust_nms_threshold(operating_config=operating_config,
                             adjust_by=-operating_config.NMS_THRESHOLD_ADJUSTBY)
        operating_config.increment_show_info_counter()
    # Take screenshot
    elif key%256 == 32: # SPACE pressed
        write_progress_image(img=img,
                             operating_config=operating_config
                            )   
  

def handle_config_key_input(img,
                            key,
                            operating_config):

        
    if key%256 == 103: # letter g, Go and process video
        print(f'Running process')
        operating_config.CONFIGURE = False
    
    
    handle_general_key_input(img=img,
                             key=key,
                             operating_config=operating_config)
    
  
def handle_processing_key_input(img,
                                key,
                                operating_config):
    if key%256 == 99: #small c
        operating_config.PROCESS_IMAGES = False
        operating_config.CONFIGURE = True

    handle_general_key_input(img=img,
                             key=key,
                             operating_config=operating_config)



def adjust_nms_threshold(operating_config,
                         adjust_by):
    
    new_threshold = round(operating_config.NMS_THRESHOLD + adjust_by,2)
    
    if new_threshold < 0: #Can't go below 0
        # Adding the operating config threshold stays if model changes
        operating_config.NMS_THRESHOLD = 0
        operating_config.modelNet.nms_threshold = 0
    elif new_threshold > 1: #Can't go above 1
        # Adding the operating config threshold stays if model changes
        operating_config.NMS_THRESHOLD = 1
        operating_config.modelNet.nms_threshold = 1
    else:
        # Adding the operating config threshold stays if model changes
        operating_config.NMS_THRESHOLD = new_threshold
        operating_config.modelNet.nms_threshold = new_threshold


def adjust_confidence_threshold(operating_config,
                                adjust_by):
    
    new_threshold = round(operating_config.CONFIDENCE_THRESHOLD + adjust_by,2)
    
    if new_threshold < 0: #Can't go below 0
        # Adding the operating config threshold stays if model changes
        operating_config.CONFIDENCE_THRESHOLD = 0
        operating_config.modelNet.confidence_threshold = 0
    elif new_threshold > 1: #Can't go above 1
        # Adding the operating config threshold stays if model changes
        operating_config.CONFIDENCE_THRESHOLD = 1
        operating_config.modelNet.confidence_threshold = 1
    else:
        # Adding the operating config threshold stays if model changes
        operating_config.CONFIDENCE_THRESHOLD = new_threshold
        operating_config.modelNet.confidence_threshold = new_threshold


def set_desired_model(operating_config,
                      increment):
    desired_model=operating_config.detection_model
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
    
    operating_config.detection_model=desired_model
    print(f'Desired model: {desired_model}')

def write_config_info(img,
                      operating_config):
    h, w, c = img.shape

    cv.putText(img, 
               f'NMS threshold: {str(operating_config.modelNet.nms_threshold)}', 
               (40, h-50), 
               operating_config.font, 
               1, 
               (100, 255, 0), 
               2, 
               cv.LINE_AA)

    cv.putText(img, 
               f'Confidence threshold: {str(operating_config.modelNet.confidence_threshold)}', 
               (40, h-90), 
               operating_config.font, 
               1, 
               (100, 255, 0), 
               2, 
               cv.LINE_AA)

    cv.putText(img, 
               f'Model: {operating_config.modelNet.model_type}', 
               (40, h-130), 
               operating_config.font, 
               1, 
               (100, 255, 0), 
               2, 
               cv.LINE_AA)
    return img


def write_row_of_info(img,
                      info,
                      rownum,
                      operating_config,
                      draw_line=False):
    colour = (100, 255, 0)
    line_width=350
    line_thickness=2
    font_thickness=2
    font_size=0.8
    if rownum==1:
        font_size=1
    cv.putText(img, 
               info, 
               (40, 60 + (45 * (rownum-1))), 
               operating_config.font, 
               font_size, 
               colour, 
               font_thickness, 
               cv.LINE_AA)
    
    
    if draw_line:
        cv.line(img, (40, 70 + (45 * (rownum-1))), (40 + line_width, 70 + (45 * (rownum-1))), colour, thickness=line_thickness)
    
    return img

def write_config_screen(img,
                        operating_config):
        
    screen_info=[]
    screen_info.append(f'Runtime Configuration')
    screen_info.append(f'')
    screen_info.append(f'Detection model:  {operating_config.detection_model} (keys: [ and ])')
    screen_info.append(f"Confidence Threshold: {operating_config.modelNet.confidence_threshold} (keys: + and -)")
    screen_info.append(f"NMS Threshold: {operating_config.modelNet.nms_threshold} (keys: ; and \')")
    screen_info.append(f'')
    screen_info.append(f'Detection:          {"On" if operating_config.MODELNET_DETECT else "Off"} (key: d)')
    screen_info.append(f'Object labels:      {"On" if operating_config.SHOW_LABELS else "Off"} (key: l)')
    screen_info.append(f'Show FPS/frames: {"On" if operating_config.SHOW_FPS else "Off"} (key: f)')
    screen_info.append(f'Display runtime:    {"On" if operating_config.SHOW_RUNTIME_CONFIG else "Off"} (key: i)')
    
    for i in range(1,len(screen_info)+1):
        draw_line=False
        if i == 1:
            draw_line=True
        
        img = write_row_of_info(img=img,
                                info=screen_info[i-1],
                                rownum=i,
                                operating_config=operating_config,
                                draw_line=draw_line)

    return img

def write_loading_model(img,
                        operating_config):
    h, w, c = img.shape

    cv.putText(img, 
               f'Initializing model: {str(operating_config.detection_model)}', 
               (300, int(h/2)-25), 
               operating_config.font, 
               1, 
               (100, 255, 0), 
               2, 
               cv.LINE_AA) 
    return img