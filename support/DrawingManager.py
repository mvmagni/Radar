from support.OperatingConfig import OperatingConfig
import cv2 as cv
from support.BoundingBox import BoundingBox
from statistics import mean
import time
from support.ModelNet import ModelNet
from support.DetectedObject import DetectedObject
from support.Hands import Hands
from support.MoveNet import MoveNet

class DrawingManager:
    
    def __init__ (self,
                  operatingConfig: OperatingConfig,
                  mpHands: Hands,
                  moveNet: MoveNet):
        self.operatingConfig = operatingConfig
        self.mpHands = mpHands
        self.moveNet = moveNet
        
    def draw_results(self,
                     img, 
                     detected_objects: list[DetectedObject], 
                     weight=1):
        
        
        if self.operatingConfig.SHOW_BOUNDING_BOXES:
            self.draw_bounding_box(img=img,
                                   detected_objects=detected_objects)
        
        if self.operatingConfig.SHOW_LABELS:
             self.show_label(img=img,
                             detected_objects=detected_objects)
        
        if self.operatingConfig.SHOW_HANDS:
            self.show_hands(img=img,
                            detected_objects=detected_objects)
        
        if self.operatingConfig.SHOW_POSE:
            self.show_pose(img=img,
                           detected_objects=detected_objects)
        
        ###########################################################
        # General visual display info
        if self.operatingConfig.SHOW_FPS:
            self.show_fps(img=img)
        
        if (self.operatingConfig.SHOW_RUNTIME_CONFIG or 
            self.operatingConfig.show_runtime_config_until_frame > self.operatingConfig.frame_counter):

            img = self.write_config_info(img=img)
        #=============================================================
        
        return img

    def show_pose(self,
                  img,
                  detected_objects: list[DetectedObject]):
        print("DrawingManager.show_pose called")
        for obj in detected_objects:
            if obj.object_type == self.operatingConfig.OBJECT_POSE:
                self.moveNet.draw_poses(img=img,
                                        pose_keypoints=obj.pose_keypoints,
                                        confidence_threshold=0.2)


    def show_hands(self,
                   img,
                   detected_objects: list[DetectedObject]):
        for obj in detected_objects:
            if obj.object_type == self.operatingConfig.OBJECT_HAND:
                self.mpHands.draw_hands(img=img,
                                        hand_landmarks=obj.hand_landmarks)
            

    def draw_bounding_box(self,
                          img,
                          detected_objects: list[DetectedObject],
                          weight=1):
        
        # Iterate through all objects and draw
        for obj in detected_objects:
            
            colour=self.get_class_colour(obj.classID)
            
            x,y = obj.bbox.get_top_left()
            w = obj.bbox.width
            h = obj.bbox.height
            
            cv.rectangle(img,(x,y), (x+w, y+h), colour, weight)
            
            line_width_max = 50
            line_width = min(int(obj.bbox.width/2 * 0.30), line_width_max)
            line_height = min(int(obj.bbox.height/2 * 0.30), line_width_max)
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


    def show_label(self,
                   img, 
                   detected_objects: list[DetectedObject]):
        
        for obj in detected_objects:
            
            if obj.confidence is None:
                confidence_label=f'{obj.className}'
            else:
                confidence_label=f'{obj.className} {int(obj.confidence*100)}%'
            
            colour = self.get_class_colour(classID=obj.classID)
            x, y = obj.bbox.get_top_left()
            cv.putText(img,
                    confidence_label,
                    (x,y-5), 
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    colour,
                    2
                    )
        return img

    def get_class_colour(self,
                         classID):
        if classID == 0: # person
            colour = (255,51,153) # colour=blue
        elif classID == 2: # car
            colour = (255,153,51) # colour=cyan
        else: 
            colour = (0,204,204) # colour=yellow
        
        return colour

    def write_config_info(self,
                          img):
        h, w, c = img.shape

        cv.putText(img, 
                f'NMS threshold: {str(self.operatingConfig.NMS_THRESHOLD)}', 
                (40, h-50), 
                self.operatingConfig.font, 
                1, 
                (100, 255, 0), 
                2, 
                cv.LINE_AA)

        cv.putText(img, 
                f'Confidence threshold: {str(self.operatingConfig.CONFIDENCE_THRESHOLD)}', 
                (40, h-90), 
                self.operatingConfig.font, 
                1, 
                (100, 255, 0), 
                2, 
                cv.LINE_AA)

        cv.putText(img, 
                f'Model: {self.operatingConfig.detection_model}', 
                (40, h-130), 
                self.operatingConfig.font, 
                1, 
                (100, 255, 0), 
                2, 
                cv.LINE_AA)
        return img

    def show_fps(self, 
                 img):
        queue_length=60
        
        if self.operatingConfig.fps_queue is None:
            self.operatingConfig.fps_queue = []
        
        # font which we will be using to display FPS
        font = cv.FONT_HERSHEY_SIMPLEX
        # time when we finish processing for this frame
        new_frame_time = time.time()
    
        # Calculating the fps
    
        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1/(new_frame_time - self.operatingConfig.prev_frame_time)
        self.operatingConfig.prev_frame_time = new_frame_time
    
        # converting the fps into integer
        fps = int(fps)
    
        # add fps
        self.operatingConfig.fps_queue.append(fps)
        
        # Remove anything above desired queue/avg length
        while len(self.operatingConfig.fps_queue) > queue_length:
            self.operatingConfig.fps_queue.pop(0)
    
        # putting the FPS count on the frame
        cv.putText(img, 
                'FPS:'+ str(int(mean((self.operatingConfig.fps_queue)))), 
                (15, 50), 
                font, 
                1.0, 
                (100, 255, 0), 
                2, 
                cv.LINE_AA)
            # putting the FPS count on the frame
        cv.putText(img, 
                f'Frame: {str(self.operatingConfig.frame_counter)}', 
                (15, 85), 
                font, 
                1.0, 
                (100, 255, 0), 
                2, 
                cv.LINE_AA)

    def write_info_bottom_left(self,
                                img,
                                info):
        h, w, c = img.shape
        # print(f'w:{w}, h:{h}')
        cv.putText(img, 
                info, 
                (40, h-50), 
                self.operatingConfig.font, 
                1.5, 
                (100, 255, 0), 
                2, 
                cv.LINE_AA)

    def write_row_of_info(self,
                            img,
                            info,
                            rownum,
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
                self.operatingConfig.font, 
                font_size, 
                colour, 
                font_thickness, 
                cv.LINE_AA)
        
        
        if draw_line:
            cv.line(img, (40, 70 + (45 * (rownum-1))), (40 + line_width, 70 + (45 * (rownum-1))), colour, thickness=line_thickness)
        
        return img

    def write_config_screen(self,
                            img,
                            modelNet:ModelNet):
            
        screen_info=[]
        screen_info.append(f'Runtime Configuration')
        screen_info.append(f'')
        screen_info.append(f'Detection model:  {self.operatingConfig.detection_model} (keys: [ and ])')
        screen_info.append(f"Confidence Threshold: {modelNet.confidence_threshold} (keys: + and -)")
        screen_info.append(f"NMS Threshold: {modelNet.nms_threshold} (keys: ; and \')")
        screen_info.append(f'')
        screen_info.append(f'Detection:          {"On" if self.operatingConfig.MODELNET_DETECT else "Off"} (key: d)')
        screen_info.append(f'Object labels:      {"On" if self.operatingConfig.SHOW_LABELS else "Off"} (key: l)')
        screen_info.append(f'Show FPS/frames: {"On" if self.operatingConfig.SHOW_FPS else "Off"} (key: f)')
        screen_info.append(f'Display runtime:    {"On" if self.operatingConfig.SHOW_RUNTIME_CONFIG else "Off"} (key: i)')
        
        for i in range(1,len(screen_info)+1):
            draw_line=False
            if i == 1:
                draw_line=True
            
            img = self.write_row_of_info(img=img,
                                    info=screen_info[i-1],
                                    rownum=i,
                                    draw_line=draw_line)

        return img

    def write_loading_model(self,
                            img):
        h, w, c = img.shape

        cv.putText(img, 
                f'Initializing model: {str(self.operatingConfig.detection_model)}', 
                (300, int(h/2)-25), 
                self.operatingConfig.font, 
                1, 
                (100, 255, 0), 
                2, 
                cv.LINE_AA) 
        return img