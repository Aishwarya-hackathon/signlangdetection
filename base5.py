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
%cd {PRETRAINED_MODEL_PATH} && tar -zxvf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'

#!mkdir {'Tensorflow\workspace\models\\'+CUSTOM_MODEL_NAME}
#!cp {PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config'} {MODEL_PATH+'/'+CUSTOM_MODEL_NAME}

#in cmd-administrator
#pip3 install --user tensorflow
#python 

import tensorflow as tf

#protoc object_detection/protos/*.proto --python_out=.
#Add the following to your ~/.bashrc :
#export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/TensorFlow/models/research/object_detection:<PATH_TO_TF>/TensorFlow/models/research:<PATH_TO_TF>/TensorFlow/models/research/slim
#!pip install -qq pycocotools
%cd D:\projects\signlangdetection\Tensorflow\models\research\
!protoc object_detection\protos\*.proto --python_out=.
#%cp object_detection/packages/tf2/setup.py .
#!python -m pip install


#!python object_detection\builders\model_builder_tf2_test.py
import tensorflow.compat.v1 as tf

from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


CONFIG_PATH = 'D:\projects\signlangdetection\Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config'

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