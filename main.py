import cv2 as cv
import os
from configparser import ConfigParser

from support.IOManager import IOManager
import support.yolo_config as yc
from support.OperatingConfig import OperatingConfig
from support.FrameManager import FrameManager

############################################################
# Load and parse configuration file
config_object=ConfigParser()
config_object.read(f'{os.getcwd()}/Radar/config.ini')

config_system=config_object['SYSTEM']
config_directories=config_object['DIRECTORY']
config_variables=config_object['VARIABLES']
#############################################################

#Configs for changing video while running
operatingConfig = OperatingConfig(image_store_dir=f'{config_directories["image_store_dir"]}',
                            resource_dir=f'{config_system["project_root_dir"]}/{config_system["resource_dir"]}',
                            model_config_dir=f'{config_system["project_root_dir"]}/{config_system["model_config_dir"]}',
                            className_file = f'{config_system["project_root_dir"]}/{config_system["className_file"]}')


frameManager = FrameManager(operatingConfig=operatingConfig)
ioManager = IOManager(operatingConfig=operatingConfig)

###############################################################################
# Config video capture. 0 is first
# Added cv.CAP_DSHOW to avoid several minute lag of opening cam on windows
# No lag opening on Linux

# Webcam
#cap = cv.VideoCapture(0,cv.CAP_DSHOW)
cap = cv.VideoCapture(1,cv.CAP_DSHOW)
CAP_WEBCAM=True

# Video file defined above
#CAP_WEBCAM=False
#cap = cv.VideoCapture(f'{config_system["project_root_dir"]}/{config_system["sample_video"]}')
#cap = cv.VideoCapture(f'd:/Radar_Support/Hand_Tracking_Full.mp4')
#cap = cv.VideoCapture(f'd:/Radar_Support/GoWeb.mp4')

# Jarvis rtsp stream
#gst = 'rtspsrc location=rtsp://172.20.0.30:8554/unicast latency=10 ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1'
#cap = cv.VideoCapture(gst,cv.CAP_GSTREAMER)

fps = cap.get(cv.CAP_PROP_FPS)
cap.set(cv.CAP_PROP_FRAME_WIDTH,1600)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,900)
#================================================================================

# Load background image
bg_img_orig = cv.imread(f'{config_system["project_root_dir"]}/{config_variables["background_image"]}')

# Run program
while operatingConfig.RUN_PROGRAM: 
    ##############################################################
    # Configure screen processing
    while operatingConfig.CONFIGURE and operatingConfig.RUN_PROGRAM:
        bg_img = bg_img_orig.copy()
        frameManager.write_config_screen(img=bg_img)
        cv.imshow(f'{config_variables["main_window_name"]}', bg_img)
        
        # Handle input for configuration
        k = cv.waitKey(0)
        ioManager.handle_config_key_input(img=bg_img,
                                          key=k)
  
    ###############################################################
    # no longer in configure screen
    # start setup for processing images
    load_bg_img = bg_img_orig.copy()
    frameManager.write_loading_model(img=load_bg_img)
    cv.imshow(f'{config_variables["main_window_name"]}', load_bg_img)
        
    # Completed config, set system to process images
    operatingConfig.PROCESS_IMAGES = True

    ###################################################################
    # Main loop for processing images
    # Image processing is based on operatingConfig paramaters
    # which can be changed at runtime
    while operatingConfig.PROCESS_IMAGES and operatingConfig.RUN_PROGRAM:
        
        success, frame = cap.read()
        if success:
            operatingConfig.frame_counter += 1
        else:
            break
        
        if CAP_WEBCAM:
            frame = cv.flip(frame, 1)
        
        # If first frame show image info for debugi
        if operatingConfig.frame_counter == 1:
            print(f'Image size: {frame.shape}')

        # Image frame will be processed based on the current configuration
        # stored in operatingConfig
        processed_frame = frameManager.process_image(img=frame)

        # Show the image
        cv.imshow(f'{config_variables["main_window_name"]}',processed_frame)

        ####################################################
        # Handle key inputs
        #
        # Esc to close, space to write a copy of the image
        # pass any keypress to be processed
        k = cv.waitKey(1)
        ioManager.handle_processing_key_input(img=frame,
                                              key=k)     
        #===================================================

#######################################################
# Exit image processing
#
# ESC has been pressed to exit. Now do cleanup
# Release the cam link
cap.release()

# Clear all the windows
cv.destroyAllWindows()
#======================================================