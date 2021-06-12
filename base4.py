import os
import sys
import numpy as np

import cv2
from PyQt5 import QtCore
from PyQt5.QtCore  import pyqtSlot
from PyQt5.QtGui import QImage , QPixmap
from PyQt5.QtWidgets import QDialog , QApplication,QMainWindow,QWidget

from PyQt5.uic import loadUi


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
        self.UiComponents() 
        
    def imgfunc(self):
        global label
        label = self.picname.currentText()
        
        os.mkdir ('D:/projects/signlangdetection/Tensorflow/workspace/images/collectedimgs//'+label)
        
        
    
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
        
        self.TEXT.setText('Kindly Press "Take Pictures !" to Capture image')
        
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
                    img_path='D:/projects/signlangdetection/Tensorflow/workspace/images/collectedimgs'
                    cv2.imwrite('D:/projects/signlangdetection/Tensorflow/workspace/images/collectedimgs/%s/%s_%s.png'%(label ,label ,self.value),frame)
    
                    self.logic= 1
                    self.TEXT.setText('Your Image has been Saved \n\nPress CLOSE to close window/camera')
                    
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