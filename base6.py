# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:00:49 2021

@author: Aishwarya
"""

WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'

CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
%cd D:\projects\signlangdetection

labels = [
    {'name':'Hello', 'id':1},
    {'name':'Bye', 'id':2},
    {'name':'Okay', 'id':3},
    {'name':'Yes', 'id':4},
    {'name':'No', 'id':5},
]
with open(ANNOTATION_PATH + '\label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')
        

import os

open(ANNOTATION_PATH + '/train.record', 'w')
open(ANNOTATION_PATH + '/test.record', 'w')

!python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x {IMAGE_PATH + '/train'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/train.record'}
!python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x {IMAGE_PATH + '/test'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/test.record'}
#%cd {PRETRAINED_MODEL_PATH} && tar -zxvf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz

!cd Tensorflow && git clone https://github.com/tensorflow/models
%cd D:\projects\signlangdetection\Tensorflow\models\research
!git clone https://github.com/waleedka/coco 
!pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
%cd D:\projects\signlangdetection\Tensorflow\models\research\coco\PythonAPI
!python setup.py build_ext --inplace
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'

#!mkdir {'Tensorflow\workspace\models\\'+CUSTOM_MODEL_NAME}
#!cp {PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config'} {MODEL_PATH+'/'+CUSTOM_MODEL_NAME}

#in cmd-administrator
#pip3 install --user tensorflow
#python 
%cd C:\additionalpkgs\
import tensorflow as tf

#protoc object_detection/protos/*.proto --python_out=.
#Add the following to your ~/.bashrc :
#export PYTHONPATH=$PYTHONPATH:TensorFlow/models/research/object_detection:TensorFlow/models/research:TensorFlow/models/research/slim

%cd D:\projects\signlangdetection\Tensorflow\models\research\
!protoc object_detection\protos\*.proto --python_out=.
!cp object_detection/packages/tf2/setup.py .
!python -m pip install


!python object_detection\builders\model_builder_tf2_test.py
#import tensorflow.compat.v1 as tf

from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'

config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
 
config

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)
    
##CHANGE TO LEN(LABELS)
len(labels)
pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']

config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)
import sys
sys.path.append("Tensorflow\\models\\research\\")
sys.path.append("Tensorflow\\models\\research\\object_detection\\utils")

%cd D:\projects\signlangdetection\Tensorflow\models\research\object_detection
!python model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=5000