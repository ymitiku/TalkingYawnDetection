import cv2
import dlib
import os
import numpy as np
import pandas as pd
import json


class DriverActionDataset(object):
    def __init__(self,dataset_dir,image_shape):
        self.dataset_dir = dataset_dir
        self.image_shape = image_shape
    def get_attribute(self,path):
        _,file_name = os.path.split(path)
        fname, _ = os.path.splitext(file_name)
        subj,gender_glasses,action_str = fname.split("-")
        gender = -1
        glasses_str = None
        if gender_glasses[:4] == "Male":
            gender = 1
            glasses_str = gender_glasses[4:]
        elif gender_glasses[:6] == "Female":
            gender = 0
            glasses_str = gender_glasses[6:]
        else:
            raise Exception("Unable to parse gender from "+str(path))
        glasses = -1
        
        if glasses_str =="NoGlasses":
            glasses = 0
        elif glasses_str  == "Glasses" or glasses_str == "SunGlasses":
            glasses = 1
        else:
            raise Exception("Unable to parse glasses information from "+str(path))
        action = -1
        if action_str=="Normal":
            action = 0
        elif action_str == "Yawning":
            action = 1
        elif action_str == "Talking":
            action = 2
        else:
            raise Exception("Unable to parse action information from " + str(path))
        output = {"Subject":subj,"Gender":gender,"Glasses":glasses,"Action":action}
        return output

    def video_to_image_sequence(self,video_path,output_path):
        print "Processing",video_path," video"
        cap = cv2.VideoCapture(video_path)
        count = 0
        while cap.isOpened():
            _,frame = cap.read()
            cv2.imwrite(os.path.join(output_path,str(count)+".jpg"),frame)
            count += 1
        cap.release()
        print "Processed",video_path," video"
    def video_sequences_to_image_sequences(self,dataset_dir,output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)  
        for video_name in os.listdir(os.path.join(dataset_dir,"Female_mirror")):
            video_path = os.path.join(dataset_dir,"Female_mirror",video_name)
            video_attr = self.get_attribute(video_path)
            _,file_name = os.path.split(video_path)
            fname, _ = os.path.splitext(file_name)
            output_dir = os.path.join(output_path,fname)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            with open(os.path.join(output_dir,"attr.json"),"w+") as attr_file:
                json.dump(video_attr,attr_file)
            self.video_to_image_sequence(video_path,output_dir)
        for video_name in os.listdir(os.path.join(dataset_dir,"Male_mirror")):
            video_path = os.path.join(dataset_dir,"Male_mirror",video_name)
            video_attr = self.get_attribute(video_path)
            _,file_name = os.path.split(video_path)
            fname, _ = os.path.splitext(file_name)
            output_dir = os.path.join(output_path,fname)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            with open(os.path.join(output_dir,"attr.json"),"w+") as attr_file:
                json.dump(video_attr,attr_file)
            self.video_to_image_sequence(video_path,output_dir)
        
    def load_dataset(self):
        pass