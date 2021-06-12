# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 18:22:22 2021

@author: Aishwarya
"""

#import dependencies
import os  #work with file paths
import uuid #name img files 
import cv2 #opencv
import time #hand movement

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QHBoxLayout, QWidget

img_path='D:/projects/signlangdetection/Tensorflow/workspace/collectedimgs'

labels = ['hello','no']

number_imgs=5

def button_pressed():

    for label in labels:
        os.mkdir ('D:/projects/signlangdetection/Tensorflow/workspace/collectedimgs//'+label)
        cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        #cap.set(cv2.CAP_PROP_BRIGHTNESS, 1)
            
        print('Collecting images for {}'.format(label))
        time.sleep(5)
        for imgnum in range(number_imgs):
            # Capture frame-by-frame
            ret, frame=cap.read()
            imgname=os.path.join(img_path, label, label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
            cv2.imwrite(imgname,frame)
            cv2.imshow(label,frame)
            time.sleep(2)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
        

    
def button_pressed2():
    print('Training done!')
    
app = QApplication([])
win = QMainWindow()
central_widget = QWidget()


button = QPushButton('Take Pictures', central_widget)
#button.setGeometry(0,0,600,600) #else buttons may overlap
#button2 = QPushButton('Train', central_widget)
layout = QHBoxLayout(central_widget)
layout.addWidget(button)
#layout.addWidget(button2)


button.clicked.connect(button_pressed)

#button.clicked.connect(button_pressed2)


win.setCentralWidget(central_widget)
win.show()
app.exit(app.exec_())