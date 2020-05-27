#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:19:50 2020

@author: shubham
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer



#####

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model

import cv2
from fr_utils import *
from inception_blocks_v2 import *

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def triplet_loss(y_true, y_pred, alpha = 0.2):
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(anchor-positive),axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor-negative),axis=-1)
    basic_loss = tf.maximum(pos_dist-neg_dist+alpha,0)
    loss = tf.reduce_sum(basic_loss)
    
    return loss

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

database = {}
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)

def who_is_it(image_path, database, model):
    encoding = img_to_encoding(image_path, model)
    min_dist = 100
    for name in database:
        dist = np.linalg.norm(encoding-database[name])
        if (dist<min_dist):
            min_dist = dist
            identity = name    
    if min_dist > 0.7:
        identity="Not in database!!  "+ str(min_dist)
        
    return min_dist, identity


########

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        min_dist,identity = who_is_it(file_path, database, FRmodel)
        result=str(identity)+ str(min_dist)
        if os.path.exists(file_path):
            os.remove(file_path)
        return result
    return None




if __name__ == '__main__':
    app.run(debug =True)
