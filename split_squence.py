import os
import shutil
import cv2
import dlib
import numpy as np
import json


def split_array(array,max_size):
    output = []
    for i in range(0,len(array),max_size):
        output+=[array[i:i+max_size]]
    return output
def copy_images(imgs_files,source_folder,dest_folder):
     for imfile in imgs_files:
         shutil.copy(os.path.join(source_folder,imfile),os.path.join(dest_folder,imfile))

def split_sequence(dataset_dir,output_dir,max_size):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    sequences = os.listdir(dataset_dir)
    for s in sequences:       
        current_path  = os.path.join(dataset_dir,s)
        s_images = os.listdir(current_path)
        s_images.sort()
        splited_seq = split_array(s_images,max_size) 
        for i in range(len(splited_seq)):
            dest_folder = os.path.join(output_dir,s+"-"+str(i))
            if not os.path.exists(dest_folder):
                os.mkdir(dest_folder)
            copy_images(splited_seq[i],current_path,dest_folder)
        print "Processed",s
def rect_to_array(rect):
    output = []
    output[0:4] = rect.left(),rect.top(),rect.right(),rect.bottom()
    return output
def track_all_faces(sequence_path,img_files,face_index,detector,predictor):
    img = cv2.imread(os.path.join(sequence_path,img_files[face_index]))
    face = detector(img)[0]
    tracker = dlib.correlation_tracker()
    win = dlib.image_window()   
    tracker.start_track(img,face)
    bounding_boxes = {}
    for i in range(face_index,-1,-1):
        img = cv2.imread(os.path.join(sequence_path,img_files[i]))
        tracker.update(img)
        tracked_face = tracker.get_position()
        bounding_boxes[img_files[i]] = rect_to_array(tracked_face)
        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(tracked_face)

    img = cv2.imread(os.path.join(sequence_path,img_files[face_index]))
    face = detector(img)[0]
    tracker.start_track(img,face)
    for i in range(face_index+1,len(img_files)):
        img = cv2.imread(os.path.join(sequence_path,img_files[i]))
        tracker.update(img)
        tracked_face = tracker.get_position()
        bounding_boxes[img_files[i]] = rect_to_array(tracked_face)
        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(tracked_face)
    return bounding_boxes

def track_face_inside_sequence(sequence_path,output_dir):
    img_files = os.listdir(sequence_path)
    img_files.sort()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    bounding_box = {}
    face_found = False
    sequence_basename = os.path.basename(sequence_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i in  range(len(img_files)):
        img = cv2.imread(os.path.join(sequence_path,img_files[i]))
        faces = detector(img)
        if len(faces)>0:
            bounding_box = track_all_faces(sequence_path, img_files,i,detector,predictor)
            with open(os.path.join(output_dir,sequence_basename)+".json","w+") as bbox_file:
                json.dump(bounding_box,bbox_file)
            face_found = True
            break
    if not face_found:
        print "No faces found inside ",sequence_path, " sequence"
def track_faces_inside_sequences(dataset_dir,output_dir):
    for seq in os.listdir(dataset_dir):
        track_face_inside_sequence(os.path.join(dataset_dir,seq),output_dir)
    
