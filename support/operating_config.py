import cv2 as cv
import os
import support.yolo_config as yc
from dataclasses import dataclass, field
from support.model_net import ModelNet
from support.Hands import Hands

@dataclass
class operatingConfig:
    #############################################
    # Init params
    # Where to store screenshots
    image_store_dir: str
    resource_dir: str
    model_config_dir: str
    className_file: str
    #=============================================    
    
    ###############################################
    # ModelNet parameters
    # Default modelnet to load
    DEFAULT_MODEL: str = yc.MODEL_YOLOV4N_800_448    
    modelNet: ModelNet = field(init=False)
    detection_model: str = field(init=False)

    # ModelNet - Confidence threshold adjustment amount used in interface
    CONFIDENCE_THRESHOLD=0.6
    CONF_THRESHOLD_ADJUSTBY=0.05
    NMS_THRESHOLD=0.3
    NMS_THRESHOLD_ADJUSTBY=0.05
    #==============================================

    #######################################################
    #Configs for changing display while running
    SHOW_FPS: bool = False
    SHOW_RUNTIME_CONFIG: bool = False
    
    SHOW_LABELS: bool = False
    
    MODELNET_DETECT: bool = False   
    SHOW_HANDS: bool = False
    #======================================================

    #######################################################
    # Control flow parameters    
    # Whether the program should RUN or not
    RUN_PROGRAM: bool = True
    
    # Show Configuration screen
    CONFIGURE: bool = True
    
    # Process video file
    PROCESS_IMAGES: bool = True
    #======================================================
    
    ###########################################################
    # Runtime variables
    # Keeps track of changes while system is running
    fps_queue: list = field(default_factory=list)
    prev_frame_time: int = 0
    frame_counter: int = 0
    #==========================================================
    
    
    ###########################################################
    # Default values for system processing
    # TODO: move defaults externally 
    
    # How many frames to show Model info on activate/switch
    # Default increment amount to show on change
    SHOW_RUNTIME_CONFIG_FRAME_INCREMENT: int = 90
    
    # Runtime value based on frame_counter, start@90
    # when switching values adds SHOW_RUNTIME_CONFIG_FRAME_INCREMENT to framecounter
    # into the variable below
    show_runtime_config_until_frame: int = 90 
    
    # Default common values
    font = cv.FONT_HERSHEY_SIMPLEX
    #=========================================================

    def __post_init__(self):
        self.detection_model = self.DEFAULT_MODEL
        self.print_environment()
        self.create_modelNet()
        self.create_hand_track()

    def increment_show_info_counter(self):
        self.show_runtime_config_until_frame = (self.frame_counter + self.SHOW_RUNTIME_CONFIG_FRAME_INCREMENT)

    def print_environment(self):
        print(f'OpenCV Version: {cv.__version__}')
        print(f'CUDA enabled devices: {cv.cuda.getCudaEnabledDeviceCount()}')
        print(f'CWD: {os.getcwd()}')
        
    def create_modelNet(self):
        self.modelNet= ModelNet(config_dir=self.model_config_dir,
                                classname_file=self.className_file,
                                model_type=self.detection_model,
                                confidence_threshold=self.CONFIDENCE_THRESHOLD,
                                nms_threshold=self.NMS_THRESHOLD
                                )
    
    def create_hand_track(self):
        self.mpHands = Hands()