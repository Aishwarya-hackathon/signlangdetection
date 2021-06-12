import os
import sys
import numpy as np

import cv2
from PyQt5 import QtCore
from PyQt5.QtCore  import pyqtSlot
from PyQt5.QtGui import QImage , QPixmap
from PyQt5.QtWidgets import QDialog , QApplication,QMainWindow,QWidget

from PyQt5.uic import loadUi

background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment_hand(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Grab the external contours for the image
    image, contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        return (thresholded, hand_segment_max_cont)
    
class base3(QDialog):
    def __init__(self):
        super(base3,self).__init__()
        loadUi('untitled.ui',self)
        self.logic = 0
        self.value = 0
        self.pressed=0
        self.SHOW.clicked.connect(self.onClicked)
        self.CLOSE.clicked.connect(self.closecamera)
        self.TEXT.setText("Kindly Press 'Show' to connect with webcam.")
        self.TakePictures.clicked.connect(self.CaptureClicked)
        #self.picname.activated.connect(self.imgfunc)
        
    #def imgfunc(self):
        #self.labels = self.picname.currentText()
        #img_path='D:/projects/signlangdetection/Tensorflow/workspace/collectedimgs'
        #os.mkdir ('D:/projects/signlangdetection/Tensorflow/workspace/collectedimgs//'+labels)
        #return labels
    @pyqtSlot()
    def onClicked(self):
        global cap
        self.TEXT.setText('Kindly Press "Take Pictures !" to Capture image')
        
        cap =cv2.VideoCapture(0,cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        
        while(cap.isOpened()):
            ret, frame=cap.read()
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()
            roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

            gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
            
            cal_accum_avg(gray_frame, accumulated_weight)
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            
            self.displayImage(frame,1)
            cv2.waitKey()
            
            if ret:
                #print('here')
                self.displayImage(frame,1)
                cv2.waitKey()
                if (self.logic==2):
                    #self.imgname=self.imgfunc
                    self.value=self.value+1
                    cv2.imwrite('D:/projects/signlangdetection/Tensorflow/workspace/collectedimgs/%s.png'%(self.value),frame)
                    #imgname=os.path.join(img_path, labels, labels+'.'+'{}.jpg'.format(str(uuid.uuid1())))
                    #cv2.imwrite('D:/projects/signlangdetection/Tensorflow/workspace/collectedimgs//%s/%s.png'%(self.labels),%(self.value),frame)
                    #cv2.imwrite(self.imgname,frame)
                    self.logic= 1
                    self.TEXT.setText('Your Image has been Saved')
                elif (self.pressed==1):
                    break
        
               
            else:
                print('not found')
                break
            
        cap.release()
        cv2.destroyAllWindows()
        window.close()
    
    def closecamera(self):
        self.pressed=1
        
    def CaptureClicked(self):
        self.logic=2
    def displayImage(self,img,window=1):
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