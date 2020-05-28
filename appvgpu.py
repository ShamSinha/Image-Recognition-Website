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

database = {}

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


def who_is_it(face_path,file_path, database, model):
    if(cv2.imread(file_path).shape[0] <= 96 or cv2.imread(file_path).shape[1] <= 96 ):
        encoding = img_to_encoding(file_path, model)
        
    else:
        encoding = img_to_encoding(face_path,model)
       
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

app.config["ALLOWED_IMAGE_EXTENSION"] = ["PNG","JPG","JPEG"]

def allowed_image(filename):
    if not "." in filename :
        return False
    ext = filename.rsplit(".",1)[1]
    
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSION"]:
        return True
    else:
        return False
    

def SaveFace(faces,image_path,basepath,personName,ext,directory):
    
    facedata = os.path.join(
                basepath, 'haarcascade_frontalface_default.xml')
    cascade = cv2.CascadeClassifier(facedata)
    img = cv2.imread(image_path)
    
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)
    
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))
        sub_face = img[y:y+h, x:x+w]
        face_path_directory = os.path.join(
            basepath, directory)
        os.chdir(face_path_directory)
        cv2.imwrite(personName +"face."+ext , sub_face)
        face_path = os.path.join(
            basepath, directory ,personName +"face."+ext )
        return face_path
 
    
def DetectFace(image_path,basepath):
    
    facedata = os.path.join(
                basepath, 'haarcascade_frontalface_default.xml')
    cascade = cv2.CascadeClassifier(facedata)
    # Read the input image
    img = cv2.imread(image_path)
    # Convert into grayscale
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)
    # Detect faces
    faces1 = cascade.detectMultiScale(miniframe)
    
    return faces1
    

@app.route('/', methods=['GET'])
def index():
    basepath = os.path.dirname(__file__)
    directory = os.path.join(basepath, 'database')
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            file_path = os.path.join(directory , filename)
            face = DetectFace(file_path,basepath)
            if len(face) == 1:
                ext = filename.rsplit(".",1)[1]
                face_path = SaveFace(face,file_path,basepath,filename,ext,'database')
                database[filename] = img_to_encoding(face_path , FRmodel)
                
                if os.path.exists(face_path):
                    os.remove(face_path)
    
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        faces = DetectFace(file_path,basepath)
        if len(faces) == 1:   #number of faces
            ext = f.filename.rsplit(".",1)[1]
            face_path = SaveFace(faces,file_path,basepath,'unknown',ext,'uploads')
            min_dist,identity = who_is_it(face_path,file_path, database, FRmodel)
            result = str(identity)+ str(min_dist)
            
        else:
            result = "Choose Single Person Face not anything else" + str(faces.shape[0])
            
        if os.path.exists(file_path):
            os.remove(file_path)
        return result
    return None

@app.route('/add', methods=['GET'])
def addDatabase():
    return render_template('addDatabase.html')

@app.route('/addStatus', methods =['GET','POST'])
def add():
    if request.method == 'POST':
        personImage = request.files['image']    # name in input tag in addDatabase.html
        personName = request.form['personName']
        personName = personName.lower()
        
        basepath = os.path.dirname(__file__)
        
        
        if not allowed_image(personImage.filename):
            result1 = "Not recognised image file extension"
            
        else:
            ext = personImage.filename.rsplit(".",1)[1]
            file_path = os.path.join(
                basepath, 'database', personName +"."+ext)
            personImage.save(file_path)
            faces = DetectFace(file_path,basepath)
            if len(faces) == 1:   #number of faces
                face_path = SaveFace(faces,file_path,basepath,personName,ext,'database')
                database[personName] = img_to_encoding(face_path , FRmodel)
                result1 = personName + " Successfully added in Database"
            elif len(faces) >1:
                result1 = "More than one face are detected"
            elif len(faces) == 0:
                result1 = "No face are detected"
            
    return render_template('addStatus.html', result = result1)
        
      

if __name__ == '__main__':
    app.run(debug =True)
