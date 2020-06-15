from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import random

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,jsonify, send_from_directory
from werkzeug.utils import secure_filename

#####

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model

import cv2
from fr_utils import *
from inception_blocks_v2 import *
from yolo_predict import *

import matplotlib.pyplot as plt

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


def who_is_it(face, database, model):
    
    encoding = img_to_encoding(face, model)
       
    min_dist = 100
    for name in database:
        l=len(database[name])
        dist=0
        for i in range(0,l):
            dist = dist+np.linalg.norm(encoding-database[name][i])/float(l)          
        if (dist<min_dist):
            min_dist = dist
            identity = name    
    if min_dist > 1:
        identity=""
        
    return min_dist, identity


########

app = Flask(__name__)
basepath = os.path.dirname(__file__)
facedata = os.path.join(basepath, 'haarcascade_frontalface_default.xml')

def returnFaces(image1):  
    cascade = cv2.CascadeClassifier(facedata)   
    gray_image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_image,1.1, 4)           
    return faces
     

@app.route('/', methods=['GET'])
def index():
    basepath = os.path.dirname(__file__)
    directory = os.path.join(basepath, 'database')
    for face_dir in os.listdir(directory):
        
        for face_file in os.listdir(os.path.join(directory,face_dir)): 
            face_path = os.path.join(directory ,face_dir, face_file)
            face=cv2.imread(face_path,1)
            face=cv2.resize(face,(96,96))
            if not (face_file.rsplit("_",1)[0] in database.keys()) :
                database[face_file.rsplit("_",1)[0]] =[]
            database[face_file.rsplit("_",1)[0]].append(img_to_encoding(face , FRmodel))        
    
    return render_template('index.html')

@app.route('/back_to_predict',methods=['GET'])
def back_to_predict():
    return render_template('index.html')


@app.route('/predict', methods=[ 'GET','POST'])
def upload():
    if request.method == 'POST':
        c=random.randint(0,1000000000)
        our_list=os.listdir(os.path.join(basepath,'yolo'))
        if len(our_list) != 0:
            filename=our_list[0]
            our_split=filename.rsplit("_",1)[1]
            our_num=our_split.rsplit(".",1)[0]           
            if os.path.exists(os.path.join(basepath,'yolo',filename)):
                os.remove(os.path.join(basepath,'yolo',filename))
            
        f = request.files['file']
        filename=secure_filename(f.filename)
        file_path = os.path.join( basepath, 'uploads',filename)
        f.save(file_path)
        img=our_result(file_path)
        #plt.imshow(img)
        new_dir = os.path.join(basepath,'yolo')
        #os.mkdir(new_dir)
        #os.chdir(new_dir)
        cv2.imwrite(os.path.join(new_dir,"temp_"+str(c)+".jpg"),img)
        img_url="yolo/temp_"+str(c)+".jpg"
        result=""
        image=cv2.imread(file_path,1)
        flag=0
        faces = returnFaces(image) 
        print (faces,flush=True) 
        if(len(faces)==0):
            flag=1
        for face in faces: 
            x, y, w, h = [ v for v in face ]
            face_image=image[y:y+h, x:x+w]
            face_image=cv2.resize(face_image,(96,96))
            min_dist,identity = who_is_it(face_image, database, FRmodel)
            result = result+str(identity)+ " "
        
        if os.path.exists(file_path):
            os.remove(file_path)
        if(flag==0 and  len(result.strip())==0):
            result="No person is in Database !!"
        return jsonify({"result": result , "our_url" : img_url })
    return None


@app.route('/add', methods=['GET','POST'])
def addDatabase():
    return render_template('addDatabase.html')

@app.route("/yolo/<temp_name>", methods=['GET','POST'])
def for_img(temp_name):
    return send_from_directory("yolo/",temp_name)
'''
@app.route('/my_json', methods=['GET'])
def Json():
    return jsonify({"result":"hi"})
'''

@app.route('/addStatus', methods =['GET','POST'])
def add():
    if request.method == 'POST':
        personImage = request.files['image']
        personName = request.form['personName']
        personName = personName.lower()
        filename=personImage.filename
        file_path = os.path.join(basepath, 'database', filename)
        personImage.save(file_path)
        image=cv2.imread(file_path,1)
        faces = returnFaces(image)
        global database
        if not (personName in database):
            new_dir = os.path.join(basepath,'database',personName)
            face=faces[0]
            x, y, w, h = [ v for v in face ]
            face_image=image[y:y+h, x:x+w]
            os.mkdir(new_dir)
            os.chdir(new_dir)
            cv2.imwrite(personName+"_"+str(1)+"."+"jpg", face_image)
            database[personName]=[]
            face_image=cv2.resize(face_image,(96,96))
            database[personName].append(img_to_encoding(face_image , FRmodel))
        else :
            face=faces[0]
            x, y, w, h = [ v for v in face ]
            face_image=image[y:y+h, x:x+w]
            new_dir = os.path.join(basepath,'database',personName)
            l=len(os.listdir(new_dir))            
            os.chdir(new_dir)
            cv2.imwrite(personName+"_"+str(l+1)+"."+"jpg", face_image)
            face_image=cv2.resize(face_image,(96,96))
            database[personName].append(img_to_encoding(face_image, FRmodel))
        if os.path.exists(file_path):
            os.remove(file_path)
        result = "Mr/Ms. "+personName.capitalize() + " was successfully added in  our Database"
        print(result) 
        return jsonify({"result": result})
    return None




@app.route('/database', methods = ['GET' , 'POST'])
def show():
    basepath = os.path.dirname(__file__)
    person_names = os.listdir(os.path.join(basepath,'database'))
    
    return render_template('showdatabase.html', person_names = person_names)
  
@app.route('/showAllInDatabase/<person>', methods = ['GET' , 'POST'])
def showAll(person):
    basepath = os.path.dirname(__file__)
    image_names = os.listdir(os.path.join(basepath, 'database',person))
    
    return render_template('person_database.html', image_names = image_names , person = person)

@app.route('/showAllInDatabase/<person>/<filename>')
def send_all_image(person,filename):
    return send_from_directory("database/"+ person, filename)



@app.route("/removed", methods=["GET", "POST"])
def remove():
    if request.method == 'POST':
        basepath = os.path.dirname(__file__)
        for key, value in request.form.items():
            if key == "personName":
                person_name = value
                for person in os.listdir(os.path.join(basepath,'database',person_name)):
                    if(os.path.exists(os.path.join(basepath,'database',person_name,person))):
                        os.remove(os.path.join(basepath,'database',person_name,person))
                os.rmdir(os.path.join(basepath,'database',person_name))
                            
                list_database = database.keys();
                list_database_to_be_removed = []
                for i in list_database:
                    if i.rsplit("_",1)[0] == person_name:
                        list_database_to_be_removed.append(i)
                        
                for j in list_database_to_be_removed:
                    del database[j]
                 
                #result = person_name.upper() + " Successfully removed from Database"
      
    basepath = os.path.dirname(__file__)
    person_names = os.listdir(os.path.join(basepath,'database'))
    
    return render_template('showdatabase.html', person_names = person_names)

if __name__ == '__main__':
    app.run(debug =True)

