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


def who_is_it(face_path, database, model):
    
    encoding = img_to_encoding(face_path, model)
       
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

def SaveFace(file_path,basepath,personName,ext,directory):
    
    facedata = os.path.join(basepath, 'haarcascade_frontalface_default.xml')
    cascade = cv2.CascadeClassifier(facedata)   
    img = cv2.imread(file_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_image,1.1, 4)
   
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))
        sub_face = img[y:y+h, x:x+w]
        face_path_directory = os.path.join(basepath, directory)
        os.chdir(face_path_directory)
        cv2.imwrite(personName+"."+ext , sub_face)
        face_path = os.path.join(basepath, directory ,personName +"."+ext )
        return face_path
     

@app.route('/', methods=['GET'])
def index():
    basepath = os.path.dirname(__file__)
    directory = os.path.join(basepath, 'database')
    for face_file in os.listdir(directory):
        face_path = os.path.join(directory , face_file)
        database[face_file.rsplit(".",1)[0]] = img_to_encoding(face_path , FRmodel)        
    
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join( basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        ext = f.filename.rsplit(".",1)[1]
        face_path = SaveFace(file_path,basepath,'unknown',ext,'uploads')
        min_dist,identity = who_is_it(face_path, database, FRmodel)
        result = str(identity)+ " "+str(min_dist)
            
            
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(face_path):
            os.remove(face_path)
        return result
    return None

@app.route('/add', methods=['GET'])
def addDatabase():
    return render_template('addDatabase.html')

@app.route('/remove', methods=['GET'])
def removeDatabase():
    return render_template('removeDatabase.html')

@app.route('/addStatus', methods =['GET','POST'])
def add():
    if request.method == 'POST':
        personImage = request.files['image']
        personName = request.form['personName']
        personName = personName.lower()
        
        basepath = os.path.dirname(__file__)
        
        ext = personImage.filename.rsplit(".",1)[1]
        file_path = os.path.join(basepath, 'database', "unknown" +"."+ext)
        personImage.save(file_path)
        face_path = SaveFace(file_path,basepath,personName,ext,'database')
        database[personName] = img_to_encoding(face_path , FRmodel)
        if os.path.exists(file_path):
            os.remove(file_path)
        result = personName + " Successfully added in Database"
            
    return render_template('addStatus.html', result = result)


@app.route('/removeStatus', methods =['GET','POST'])
def remove():
    if request.method == 'POST':
        personName = request.form['personName']
        personName = personName.lower()
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'database', personName +"."+"jpg")
        if os.path.exists(file_path):
            os.remove(file_path)
            del database[personName]
            result = personName + " Successfully removed from Database"
        else:
            result = "The person is not in Database!!"
            
        
            
    return render_template('addStatus.html', result = result)
        
      

if __name__ == '__main__':
    app.run(debug =True)
