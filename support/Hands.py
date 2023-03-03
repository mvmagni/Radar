import cv2
import mediapipe as mp
from support.DrawingManager import DrawingManager
from support.BoundingBox import BoundingBox

class Hands:

    def __init__(self, 
                 drawingManager: DrawingManager,
                 static_image_mode=False, 
                 max_num_hands=10, 
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5
                 ):
        """
        :param static_image_mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.drawingManager = drawingManager
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.static_image_mode, 
                                        max_num_hands=self.max_num_hands,
                                        min_detection_confidence=self.min_detection_confidence,
                                        min_tracking_confidence=self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, 
                  img, 
                  draw=True, 
                  flipType=True):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #improves performance to set writeable to pass by reference
        imgRGB.flags.writeable = False
        self.results = self.hands.process(imgRGB)
        imgRGB.flags.writeable = True
        
        
        allHands = []

        h, w, c = img.shape

        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}

                ## lmList
                mylmList = []
                xList = []
                yList = []

                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## Calculate bounding box info
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin

                bbox = BoundingBox(top_left_x = xmin,
                                   top_left_y = ymin,
                                   width = boxW,
                                   height = boxH)

                myHand["lmList"] = mylmList

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, 
                                               handLms,
                                               self.mpHands.HAND_CONNECTIONS,
                                               self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                               self.mp_drawing_styles.get_default_hand_connections_style())
                    
                    self.drawingManager.show_bounding_box_modelNet(img=img,
                                         bbox=bbox,
                                         classID=None,
                                         class_name=myHand["type"],
                                         confidence=None,
                                         show_labels=True,
                                         weight=1)                    
                    
        return allHands, img

