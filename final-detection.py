# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:02:48 2021

@author: Aishwarya
"""

#print("""python {}/research/object_detection/model_main_tf2.py --model_dir={}/{} --pipeline_config_path={}/{}/
#      pipeline.config --num_train_steps=5000""".format
#     (APIMODEL_PATH, MODEL_PATH,CUSTOM_MODEL_NAME,MODEL_PATH,CUSTOM_MODEL_NAME))


WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'

CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'



labels = [
    {'name':'Hello', 'id':1},
    {'name':'Bye', 'id':2},
    {'name':'Okay', 'id':3},
    {'name':'Yes', 'id':4},
    {'name':'No', 'id':5},
]



import tensorflow as tf
import object_detection
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'



# Load Train Model From Checkpoint
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()


#Detect in Real-Time
import cv2 
import numpy as np
import sys
from PyQt5 import QtCore
from PyQt5.QtCore  import pyqtSlot
from PyQt5.QtGui import QImage , QPixmap
from PyQt5.QtWidgets import QDialog , QApplication,QMainWindow,QWidget

from PyQt5.uic import loadUi


category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')
# Setup capture
#cap = cv2.VideoCapture(0)
#width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

class base3(QDialog):
    
    def __init__(self):
        super(base3,self).__init__()
        loadUi('untitled.ui',self)
        self.logic = 0
        self.value = 0
        self.pressed=0
        self.SHOW.clicked.connect(self.onClicked)
        self.CLOSE.clicked.connect(self.closecamera)
        self.TEXT.setText("First select the folder/action name \n\nKindly Press 'Show' to connect with webcam.")
        self.TakePictures.clicked.connect(self.CaptureClicked)
        self.TEST.clicked.connect(self.TestImage)
        self.UiComponents() 
        
     
    def detect_fn(self,image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections
        
    def imgfunc(self):
        global label
        label = self.picname.currentText()
        
        os.mkdir ('D:/projects/signlangdetection/collectedimgs//'+label)
        
        
    
    def UiComponents(self): 
  
        
        labels = [' ','Hello','No','Yes','Okay','Bye','Happy']
  
        # making it editable 
        self.picname.setEditable(True) 
  
        # adding list of items to combo box 
        self.picname.addItems(labels) 
  
        # adding action to combo box 
        self.picname.activated.connect(self.imgfunc) 
  
    
    @pyqtSlot()
    def onClicked(self):
        
        self.TEXT.setText('DATASET\n1)Kindly Press "Show" to connect with webcam \n2)Kindly Press "Take Pictures !" to Capture image \n\nTEST \nPress "Test" to detect your signs !')
        #Setup to Capture
        cap =cv2.VideoCapture(0,cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        
        
        while(cap.isOpened()):
            ret, frame=cap.read()
            self.displayImage(frame,1)
            cv2.waitKey()
            
            if ret:
                #print('here')
                self.displayImage(frame,1)
                cv2.waitKey()
                
            
                if (self.logic==2):
                   
                    self.value=self.value+1
                    img_path='D:/projects/signlangdetection/collectedimgs'
                    cv2.imwrite('D:/projects/signlangdetection/collectedimgs/%s/%s_%s.png'%(label ,label ,self.value),frame)
    
                    self.logic= 1
                    self.TEXT.setText('Your Image has been Saved \n\nPress CLOSE to close window/camera')
                    
                if (self.logic==3):
                    self.TestCamera()
                    self.logic= 1
                    
                    
                elif (self.pressed==1):
                    break
               
            else:
                print('not found')
                break
               
        cap.release()
        cv2.destroyAllWindows()
        window.close()
        
    def TestCamera(self):

        cap =cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while True: 
            ret, frame = cap.read()
            image_np = np.array(frame)
            
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = self.detect_fn(input_tensor)
            
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
        
            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
            label_id_offset = 1
            image_np_with_detections = image_np.copy()
        
            viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=5,
                        min_score_thresh=.5,
                        agnostic_mode=False)
        
            cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
                

        
    def closecamera(self):
        self.pressed=1
        
    def CaptureClicked(self):
        self.logic=2
        
    def TestImage(self):
        self.logic=3
        
    def displayImage(self,img,window):
        qformat=QImage.Format_Indexed8
    
        if len(img.shape)==3:
            if(img.shape[2])==4:
                qformat=QImage.Format_RGBA888
            else:
                qformat=QImage.Format_RGB888
        img = QImage(img,img.shape[1],img.shape[0],qformat)
        img = img.rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(img))
        self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            
    


app =  QApplication(sys.argv)
window=base3()
window.setWindowTitle('Sign Language Recognition')
window.show()
sys.exit(app.exec_())