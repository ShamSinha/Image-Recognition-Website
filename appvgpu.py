#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:19:50 2020
@author: shubham
"""

from __future__ import division, print_function
# coding=utf-8

import os

import numpy as np
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



# Flask utils
from flask import Flask, redirect, url_for, request, render_template , send_from_directory
from werkzeug.utils import secure_filename

#####

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


def who_is_it(img3 , database, model):
    encoding = img_to_encoding(img3,model)
       
    min_dist = 100
    for name in database:
        dist = np.linalg.norm(encoding-database[name])
        if (dist<min_dist):
            min_dist = dist
            identity = name    
    if min_dist > 0.72:
        identity=" not present in database!! "
        
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
    

def SaveFace(do_predict,img,faces,directory,personName,count,ext):
    
    ### do_predict(boolean) = to recognise a person for a given image
    ### img = cv2.imread(given_image)
    ### faces = faces detected from the classifier 
    ### directory = save cropped face from given image in this directory
    ### personName + "|" + str(count+1) = entry in database as this name
    ### count = number of images present of personName guy
    
    basepath = os.path.dirname(__file__)
    
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))      ################################
        sub_face = img[y:y+h, x:x+w]
        face_path_directory = os.path.join(
            basepath, directory)
        os.chdir(face_path_directory)  # move into directory inside 
        
        image_name = personName.lower() +"."+ ext  
        
        cv2.imwrite(image_name, sub_face)
        face_path = os.path.join(basepath, directory, image_name )  # directory = FACES 
        
        img2 = cv2.imread(face_path, 1)
        
        img3 = cv2.resize(img2, (96,96))
                
        # if personName == 'unknown' it means we are only predicting test not adding in after prediction
        if(do_predict) :      # this is true if we want to add image in database from website as well as when we want to predict
            min_dist,identity = who_is_it(img3, database, FRmodel)
            if min_dist <= 0.72 and personName == 'unknown':
                result = "He is "+ identity + " " + "with min_dist: " + str(min_dist)
                print(result)
            if min_dist > 0.72 and personName == 'unknown':
                result = "Face Recognition System is not able to recognise may be"+ identity + "with min_dist: " + str(min_dist)
                print(result)
            if min_dist <= 0.72 and personName != 'unknown' :
                person_name_from_database = identity.rsplit("|",1)[0]
                face_match_directory = os.path.join(basepath, 'database', person_name_from_database)
                os.chdir(face_match_directory)  # move into directory inside 
                no_of_same_person_image = len(os.listdir('.'))   # number of same person image files in his/her directory
                cv2.imwrite(str(no_of_same_person_image + 1)+"."+ext,img)
                database[person_name_from_database + "|" + str(no_of_same_person_image + 1)] = img_to_encoding(img3 , FRmodel)
                result = "Successfully Added in database as existing person : " + person_name_from_database + "|" + str(no_of_same_person_image + 1)
                if os.path.exists(face_path):  # Now we remove the redundant cropped face from subdir 
                    os.remove(face_path)
               
            elif min_dist > 0.72 and personName != 'unknown':
                new_dir = os.path.join(basepath,'database',personName)
                if not os.path.isdir(new_dir):
                    os.mkdir(new_dir)
                    os.chdir(new_dir)
                    cv2.imwrite(str(1)+"."+ ext , img)
                    os.chdir(os.path.join(basepath,'FACES'))
                    cv2.imwrite(personName+"."+ext,img3)
                    database[personName.lower() + "|" + str(1)] = img_to_encoding(img3 , FRmodel)
                    result = "Successfully added in database as new person : " + personName.lower() + "|" + str(1) + " and new directory: " + personName + " are created in database directory"
                    
                else:
                    no_of_same_person_image = len(os.listdir(new_dir))
                    os.chdir(new_dir)
                    cv2.imwrite(str(no_of_same_person_image + 1)+"."+ext,img)
                    database[personName.lower() + "|" + str(no_of_same_person_image + 1)] = img_to_encoding(img3 , FRmodel)
                    result = "Successfully added in database as existing person: " + personName.lower() + "|" + str(no_of_same_person_image + 1)
                    if len(os.listdir(new_dir)) > 1 :
                        if os.path.exists(face_path):  # Now we remove the redundant cropped face from subdir 
                            os.remove(face_path)
                               
        else:       # this is true for images present in database before starting the flask app
            database[personName.lower() + "|" + str(count)] = img_to_encoding(img3 , FRmodel)
            result = "Successfully added in database as new person : " + personName.lower() + "|" + str(count) 
           
    return result
 
def DetectFace(image_path):
    
    basepath = os.path.dirname(__file__)
    
    facedata = os.path.join(
                basepath, 'haarcascade_frontalface_default.xml')
    print(facedata)
    cascade = cv2.CascadeClassifier(facedata)
    # Read the input image
    img = cv2.imread(image_path)
    print(image_path)
    # Convert into grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Applying the haar classifier to detect faces
    faces1 = cascade.detectMultiScale(gray_image,1.1, 4)
       
    return faces1
    

@app.route('/', methods=['GET'])
def index():
    basepath = os.path.dirname(__file__)
    
    directory = os.path.join(basepath, 'database')  # database path
    print(directory)
    
    subdirs = os.listdir(directory)
    print(subdirs)
    for name in subdirs:  #iterate over sub-directories present in database dir 
        personDir_path = os.path.join(directory, name)  # filepath for sub-dir
        print(personDir_path)
        count = 0 ;
        files = os.listdir(personDir_path)
        print(files)
        for person in files:  # iterate over every image file present in sub-dir
            print(person)
            if person.endswith(".png") or person.endswith(".jpg") or person.endswith(".jpeg"): 
                
                image_path = os.path.join(personDir_path , person)  # image_path in sub-dir
                print(image_path)
               
                face = DetectFace(image_path)  # detect face in image
                
                if len(face) == 1:   # if image satisfy per image one face then it get added in database as encodings
                    count = count+1
                    ext = person.rsplit(".",1)[1]
                    img = cv2.imread(image_path)
                    personName = name
                    result2 = SaveFace(False,img,face,'FACES',personName,count,ext) # extracted cropped face from image get saved in subdir
    
    print(database.keys())
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        print(file_path)
        f.save(file_path)
        faces = DetectFace(file_path)
        if len(faces) == 1:   #number of faces
            ext = f.filename.rsplit(".",1)[1]
            img = cv2.imread(file_path)
            result3 = SaveFace(True,img,faces,'uploads','unknown',00,ext)
     
        else:
            result3 = "Choose Single Person Face not anything else" 
            
        if os.path.exists(file_path):
            os.remove(file_path)
        
        print(result3)
        return result3
    return None

@app.route('/add', methods=['GET'])
def addDatabase():
    return render_template('addDatabase.html')

@app.route('/addStatus', methods =['GET','POST'])
def add():
    if request.method == 'POST':
        personImage = request.files['image']    # name in input tag in addDatabase.html
        person_name = request.form['personName']
        
        basepath = os.path.dirname(__file__)
        
        if not allowed_image(personImage.filename):
            result1 = "Not recognised image file extension"
            
        else:
            ext = personImage.filename.rsplit(".",1)[1]
            file_path = os.path.join(basepath, 'uploads', person_name +"."+ext)
            personImage.save(file_path)
            faces = DetectFace(file_path)
            if len(faces) == 1:   #number of faces
                img = cv2.imread(file_path)
                
                result1 = SaveFace(True,img,faces,'uploads',person_name,1,ext)
            
            elif len(faces) >1:
                result1 = "More than one face are detected"
            elif len(faces) == 0:
                result1 = "No face are detected"
            
    return render_template('addStatus.html', result = result1)


@app.route("/removed", methods=["GET", "POST"])
def remove():
    if request.method == 'POST':
        basepath = os.path.dirname(__file__)
        for key, value in request.form.items():
            if key == "personName":
                person_name = value
                print(person_name)
                print("key: {0}, value: {1}".format(key, value))
                #person_Dir = os.path.join(basepath,'database',person_name)
                for person in os.listdir(os.path.join(basepath,'FACES')):
                    print(person)
                    print(os.path.join(basepath,'FACES',person))
                    print(person.rsplit(".",1)[0])
                    if person.rsplit(".",1)[0] == person_name :
                        if(os.path.exists(os.path.join(basepath,'FACES',person))):
                            print("yes exists")
                            os.remove(os.path.join(basepath,'FACES',person))
            
                
                list_database = database.keys();
                list_database_to_be_removed = []
                for i in list_database:
                    if i.rsplit("|",1)[0] == person_name:
                        list_database_to_be_removed.append(i)
                        
                for j in list_database_to_be_removed:
                    del database[j]
                 
                result = person_name.upper() + " Successfully removed from Database"
        print(database.keys())
    return render_template('removeStatus.html', result = result)
        
        
@app.route('/database', methods = ['GET' , 'POST'])
def show():
    basepath = os.path.dirname(__file__)
    person_names = os.listdir(os.path.join(basepath, 'FACES'))
    
    return render_template('showdatabase.html', person_names = person_names)
  

@app.route('/database/<filename>')
def send_image(filename):
    return send_from_directory("FACES", filename)


@app.route('/showAllInDatabase/<person>', methods = ['GET' , 'POST'])
def showAll(person):
    basepath = os.path.dirname(__file__)
    image_names = os.listdir(os.path.join(basepath, 'database',person))
    
    return render_template('person_database.html', image_names = image_names , person = person)

@app.route('/showAllInDatabase/<person>/<filename>')
def send_all_image(person,filename):
    return send_from_directory("database/"+ person, filename)



if __name__ == '__main__':
    app.run(debug =True)
