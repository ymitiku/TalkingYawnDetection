import cv2
import dlib
import os
import numpy as np
import pandas as pd
import json
from threading import Thread
from sklearn.model_selection import train_test_split


class DriverActionDataset(object):
    def __init__(self,dataset_dir,bounding_box_dir,image_shape,max_sequence_length):
        self.dataset_dir = dataset_dir
        self.bounding_box_dir = bounding_box_dir
        self.image_shape = image_shape
        self.dataset_loaded = False
        self.max_sequence_length = max_sequence_length
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    def get_attribute(self,folder_name):
        subj,gender_glasses,action_str,_ = folder_name.split("-")
        gender = -1
        glasses_str = None
        if gender_glasses[:4].lower() == "male":
            gender = 1
            glasses_str = gender_glasses[4:]
        elif gender_glasses[:6].lower() == "female":
            gender = 0
            glasses_str = gender_glasses[6:]
        else:
            raise Exception("Unable to parse gender from "+str(folder_name))
        glasses = -1
        
        if glasses_str[:9].lower() =="noglasses":
            glasses = 0
        elif glasses_str[:7].lower()  == "glasses" or glasses_str[:10].lower() == "sunglasses":
            glasses = 1
        else:
            raise Exception("Unable to parse glasses information from "+str(folder_name))
        
        actions_str = action_str.split("&")
        for i in range(len(actions_str)):
            if actions_str[i].lower()=="normal":
                action = 0
            elif actions_str[i].lower() == "yawning":
                action = 0
            elif actions_str[i].lower() == "talking":
                action = 1
                break
            else:
                raise Exception("Unable to parse action information from " + str(folder_name))

        
        output = {"Subject":subj,"Gender":gender,"Glasses":glasses,"Action":action}
        
        return output
    def get_dlib_points(self,image,face,predictor):
        shape = predictor(image,face)
        dlib_points = np.zeros((68,2))
        for i,part in enumerate(shape.parts()):
            dlib_points[i] = [part.x,part.y]
        return dlib_points
    def get_right_eye_attributes(self,image,dlib_points):
        right_eye_top = int(max(dlib_points[19][1]-5,0))
        right_eye_left = int(max(dlib_points[17][0]-5,0))
        right_eye_right  = int(min(dlib_points[21][0]+5,image.shape[1]))
        right_eye_bottom = int(min(dlib_points[41][1]+5,image.shape[0]))

        right_eye = image[right_eye_top:right_eye_bottom,right_eye_left:right_eye_right]

        # r_left_corner_top   = int(max(dlib_points[19][1],0))
        # r_left_corner_left  = int(max(dlib_points[19][0],0))
        # r_left_corner_right = int(min(dlib_points[27][0],image.shape[1]))
        # r_left_corner_bottom = int(min(dlib_points[41][1],image.shape[0]))

        # right_eye_left_corner = image[r_left_corner_top:r_left_corner_bottom,r_left_corner_left:r_left_corner_right]

        # r_right_corner_top   = int(max(dlib_points[19][1],0))
        # r_right_corner_left  = int(max(dlib_points[17][0]-5,0))
        # r_right_corner_right = int(min(dlib_points[19][0],image.shape[1]))
        # r_right_corner_bottom = int(min(dlib_points[41][1]+5,image.shape[0]))

        # right_eye_right_corner = image[r_right_corner_top:r_right_corner_bottom, r_right_corner_left:r_right_corner_right]
        
        right_eye = self.resize_to_output_shape(right_eye)
        # right_eye_left_corner = self.resize_to_output_shape(right_eye_left_corner)
        # right_eye_right_corner = self.resize_to_output_shape(right_eye_right_corner)

        # return right_eye,right_eye_left_corner,right_eye_right_corner
        return right_eye

    def get_left_eye_attributes(self,image,dlib_points):
        left_eye_top = int(max(dlib_points[24][1]-5,0))
        left_eye_left = int(max(dlib_points[22][0]-5,0))
        left_eye_right  = int(min(dlib_points[26][0]+5,image.shape[1]))
        left_eye_bottom = int(min(dlib_points[46][1]+5,image.shape[0]))

        left_eye = image[left_eye_top:left_eye_bottom,left_eye_left:left_eye_right]

        # l_left_corner_top   = int(max(dlib_points[24][1],0))
        # l_left_corner_left  = int(max(dlib_points[24][0],0))
        # l_left_corner_right = int(min(dlib_points[26][0],image.shape[1]))
        # l_left_corner_bottom = int(min(dlib_points[46][1],image.shape[0]))

        # left_eye_left_corner = image[l_left_corner_top:l_left_corner_bottom,l_left_corner_left:l_left_corner_right]

        # l_right_corner_top   = int(max(dlib_points[24][1],0))
        # l_right_corner_left  = int(max(dlib_points[27][0],0))
        # l_right_corner_right = int(min(dlib_points[24][0],image.shape[1]))
        # l_right_corner_bottom = int(min(dlib_points[46][1],image.shape[0]))
        
        # left_eye_right_corner = image[l_right_corner_top:l_right_corner_bottom, l_right_corner_left:l_right_corner_right]
        
        left_eye = self.resize_to_output_shape(left_eye)
        # left_eye_left_corner = self.resize_to_output_shape(left_eye_left_corner)
        # left_eye_right_corner = self.resize_to_output_shape(left_eye_right_corner)
        
        # return left_eye,left_eye_left_corner,left_eye_right_corner 
        return left_eye
    def resize_to_output_shape(self,image):
        if image is None:
            return None
        try:
            img = cv2.resize(image,(self.image_shape[0],self.image_shape[1]))
        except:
            print "img.shape",image.shape
            return None
        return img
    def get_nose_attributes(self,image,dlib_points):
        nose_top = int(max(dlib_points[27][1]-5,0))
        nose_left = int(max(dlib_points[31][0]-5,0))
        nose_right  = int(min(dlib_points[35][0]+5,image.shape[1]))
        nose_bottom = int(min(dlib_points[33][1]+5,image.shape[0]))

        nose = image[nose_top:nose_bottom,nose_left:nose_right]

        # nose_left_corner_top   = int(max(dlib_points[27][1],0))
        # nose_left_corner_left  = int(max(dlib_points[27][0],0))
        # nose_left_corner_right = int(min(dlib_points[42][0],image.shape[1]))
        # nose_left_corner_bottom = int(min(dlib_points[33][1],image.shape[0]))

        # nose_left_corner = image[nose_left_corner_top:nose_left_corner_bottom,nose_left_corner_left:nose_left_corner_right]

        # nose_right_corner_top   = int(max(dlib_points[27][1],0))
        # nose_right_corner_left  = int(max(dlib_points[39][0],0))
        # nose_right_corner_right = int(min(dlib_points[27][0],image.shape[1]))
        # nose_right_corner_bottom = int(min(dlib_points[33][1],image.shape[0]))
        
        # nose_right_corner = image[nose_right_corner_top:nose_right_corner_bottom, nose_right_corner_left:nose_right_corner_right]
        
     

        nose = self.resize_to_output_shape(nose)
        # nose_left_corner = self.resize_to_output_shape(nose_left_corner)

        # nose_right_corner = self.resize_to_output_shape(nose_right_corner)


        # return nose,nose_left_corner,nose_right_corner
        return nose
    def get_bounding_boxes(self,sequence_path):
        _,sequence_name = os.path.split(sequence_path)
        org_squence_name = "-".join(sequence_name.split("-")[:3])
        bbox_file_path = os.path.join(self.bounding_box_dir,org_squence_name+".json")
        with open(bbox_file_path,"r") as bbox_file:
            bboxes = json.load(bbox_file)
            if bboxes is None or len(bboxes)==0:
                raise Exception("No bounding box for sequence:"+sequence_path)
            else:
                return bboxes
    def get_mouth_attributes(self,image,dlib_points):
        mouth_top = int(max(dlib_points[50][1]-5,0))
        mouth_left = int(max(dlib_points[48][0]-5,0))
        mouth_right  = int(min(dlib_points[54][0]+5,image.shape[1]))
        mouth_bottom = int(min(dlib_points[57][1]+5,image.shape[0]))

        mouth = image[mouth_top:mouth_bottom,mouth_left:mouth_right]

        # mouth_left_corner_top   = int(max(dlib_points[52][1],0))
        # mouth_left_corner_left  = int(max(dlib_points[51][0],0))
        # mouth_left_corner_right = int(min(dlib_points[54][0]+5,image.shape[1]))
        # mouth_left_corner_bottom = int(min(dlib_points[57][1],image.shape[0]))

        # mouth_left_corner = image[mouth_left_corner_top:mouth_left_corner_bottom,mouth_left_corner_left:mouth_left_corner_right]

        # mouth_right_corner_top   = int(max(dlib_points[52][1],0))
        # mouth_right_corner_left  = int(max(dlib_points[48][0],0))
        # mouth_right_corner_right = int(min(dlib_points[57][0],image.shape[1]))
        # mouth_right_corner_bottom = int(min(dlib_points[57][1],image.shape[0]))
        
        # mouth_right_corner = image[mouth_right_corner_top:mouth_right_corner_bottom, mouth_right_corner_left:mouth_right_corner_right]
        
        # mouth_top_corner_top   = int(max(dlib_points[50][1],0))
        # mouth_top_corner_left  = int(max(dlib_points[48][0],0))
        # mouth_top_corner_right = int(min(dlib_points[54][0],image.shape[1]))
        # mouth_top_corner_bottom = int(min(dlib_points[48][1],image.shape[0]))
        
        # mouth_top_corner = image[mouth_top_corner_top:mouth_top_corner_bottom, mouth_top_corner_left:mouth_top_corner_right]
        
        # mouth_bottom_corner_top   = int(max(dlib_points[48][1],0))
        # mouth_bottom_corner_left  = int(max(dlib_points[48][0],0))
        # mouth_bottom_corner_right = int(min(dlib_points[54][0],image.shape[1]))
        # mouth_bottom_corner_bottom = int(min(dlib_points[57][1],image.shape[0]))
        
        # mouth_bottom_corner = image[mouth_bottom_corner_top:mouth_bottom_corner_bottom, mouth_bottom_corner_left:mouth_bottom_corner_right]
        

        mouth = self.resize_to_output_shape(mouth)
        # mouth_left_corner = self.resize_to_output_shape(mouth_left_corner)
        # mouth_right_corner = self.resize_to_output_shape(mouth_right_corner)
        # mouth_top_corner = self.resize_to_output_shape(mouth_top_corner)
        # mouth_bottom_corner = self.resize_to_output_shape(mouth_bottom_corner)

        

        # return mouth,mouth_left_corner,mouth_right_corner,mouth_top_corner,mouth_bottom_corner
        return mouth
   
    def get_face_attributes(self,image,face,predictor):
        face_image =image[ int(max(0,face.top())):int(min(image.shape[0],face.bottom())),
                     int(max(0,face.left())):int(min(image.shape[1],face.right()))   
                    ]
        face_image = cv2.resize(face_image,(self.image_shape[0],self.image_shape[1]))

        dlib_points = self.get_dlib_points(image,face,self.predictor)
        right_eye = self.get_right_eye_attributes(image,dlib_points)
        left_eye = self.get_left_eye_attributes(image,dlib_points)
        nose = self.get_nose_attributes(image,dlib_points)
        mouth = self.get_mouth_attributes(image,dlib_points)
        output = {"face_image":face_image,"right_eye":right_eye,"left_eye":left_eye,
                    "mouth":mouth,"nose":nose
                    }
        # right_eye,right_eye_left_corner,right_eye_right_corner  = self.get_right_eye_attributes(image,dlib_points)
        # left_eye,left_eye_left_corner,left_eye_right_corner  = self.get_left_eye_attributes(image,dlib_points)
        # nose,nose_right_corner,nose_left_corner = self.get_nose_attributes(image,dlib_points)
        # mouth,mouth_left_corner,mouth_right_corner,mouth_top_corner,mouth_bottom_corner = self.get_mouth_attributes(image,dlib_points)
        # output = {"face_image":face_image,"right_eye":right_eye,"left_eye":left_eye,
        #             "mouth":mouth,"nose":nose,"left_eye_right_corner":left_eye_right_corner,
        #             "left_eye_left_corner":left_eye_left_corner,"right_eye_right_corner":right_eye_right_corner,
        #             "right_eye_left_corner":right_eye_left_corner,"nose_right_corner":nose_right_corner,
        #             "nose_left_corner":nose_left_corner,"mouth_left_corner":mouth_left_corner,
        #             "mouth_right_corner":mouth_right_corner,"mouth_top_corner":mouth_top_corner,
        #             "mouth_bottom_corner":mouth_bottom_corner
        #             }
        return output
    
    
    def load_image_sequence(self,path,detector,predictor,verbose=False):
        if verbose:
            print "loading",path
        imgs_files = os.listdir(path)
        imgs_files.sort()
        output_faces = np.zeros((self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        output_right_eyes = np.zeros((self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        output_left_eyes = np.zeros((self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        output_mouths = np.zeros((self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        output_noses = np.zeros((self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        # output_left_eye_right_corners = np.zeros((self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        # output_left_eye_left_corners = np.zeros((self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        # output_right_eye_right_corners = np.zeros((self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        # output_right_eye_left_corners = np.zeros((self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        # output_nose_right_corners = np.zeros((self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        # output_nose_left_corners = np.zeros((self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        # output_mouth_left_corners = np.zeros((self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        # output_mouth_right_corners = np.zeros((self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        # output_mouth_top_corners = np.zeros((self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        # output_mouth_bottom_corners = np.zeros((self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        
    
        bounding_boxes = self.get_bounding_boxes(path)

        for i in range(len(imgs_files)):
            img = cv2.imread(os.path.join(path,imgs_files[i]))
            if not (img is None):
                face = bounding_boxes(imgs_files[i])
                # face_image =img[ max(0,face.top()):min(img.shape[0],face.bottom()),
                #                  max(0,face.left()):min(img.shape[1],face.right())   
                #                 ]
                # [right_eye,left_eye,mouth,nose,left_eye_corners,right_eye_corners,nose_corners,mouth_corners]

                attrs = self.get_face_attributes(img, face,self.predictor)
                output_faces[i] = attrs["face_image"]
                output_right_eyes[i] = attrs["right_eye"]
                output_left_eyes[i] = attrs["left_eye"]
                output_noses[i] = attrs["nose"]
                output_mouths[i] = attrs["mouth"]
                # output_left_eye_right_corners[i] = attrs["left_eye_right_corner"]
                # output_left_eye_left_corners[i] = attrs["left_eye_left_corner"]
                # output_right_eye_right_corners[i] = attrs["right_eye_right_corner"]
                # output_right_eye_left_corners[i] = attrs["right_eye_left_corner"]
                # output_nose_right_corners[i] = attrs["nose_right_corner"]
                # output_nose_left_corners[i] = attrs["nose_left_corner"]
                # output_mouth_right_corners[i] = attrs["mouth_right_corner"]
                # output_mouth_left_corners[i] = attrs["mouth_left_corner"]
                # output_mouth_top_corners[i] = attrs["mouth_top_corner"]
                # output_mouth_bottom_corners[i] = attrs["mouth_bottom_corner"]
                    

            else:
                if verbose:
                    print ("Unable to read image from ",os.path.join(path,imgs_files[i]))
        if verbose:
            print "loaded",path
        return output_faces,output_left_eyes,output_right_eyes,output_noses,output_mouths
        # return output_faces,output_left_eyes,output_right_eyes,output_noses,output_mouths,\
        #     output_left_eye_right_corners,output_left_eye_right_corners,output_right_eye_left_corners,\
        #     output_right_eye_right_corners,output_nose_left_corners,output_nose_right_corners,\
        #     output_mouth_left_corners,output_mouth_right_corners,output_mouth_top_corners,output_mouth_bottom_corners
    def get_is_talking(self,folder_name):
        if folder_name.lower().count("talking")>0:
            return 1
        else:
            return 0
    def load_dataset(self):
        sequences = os.listdir(self.dataset_dir)

        self.train_sequences,test_sequences = train_test_split(sequences,test_size=0.05)
        self.train_sequences  =  np.array(self.train_sequences)
        # num_train_sequences  = len(train_sequences)
        num_test_sequences  = len(test_sequences)
        
        # self.face_image_train_sequences = np.zeros((num_train_sequences,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        # self.left_eye_image_train_sequences = np.zeros((num_train_sequences,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        # self.right_eye_image_train_sequences = np.zeros((num_train_sequences,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        # self.nose_image_train_sequences = np.zeros((num_train_sequences,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        # self.mouth_image_train_sequences = np.zeros((num_train_sequences,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        # self.talking_train = np.zeros((num_train_sequences,))


        # for i in range(len(train_sequences)):
        #     self.face_image_train_sequences[i],self.left_eye_image_train_sequences[i],\
        #         self.right_eye_image_train_sequences[i],self.nose_image_train_sequences[i],\
        #         self.mouth_image_train_sequences[i] = self.load_image_sequence(os.path.join(\
        #         self.dataset_dir,train_sequences[i]),detector,predictor)
        #     self.talking_train[i] = self.get_is_talking(train_sequences[i])


        self.face_image_test_sequences = np.zeros((num_test_sequences,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        self.left_eye_image_test_sequences = np.zeros((num_test_sequences,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        self.right_eye_image_test_sequences = np.zeros((num_test_sequences,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        self.nose_image_test_sequences = np.zeros((num_test_sequences,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        self.mouth_image_test_sequences = np.zeros((num_test_sequences,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        self.talking_test = np.zeros((num_test_sequences,))


        for i in range(len(test_sequences)):
            self.face_image_test_sequences[i],self.left_eye_image_test_sequences[i],self.right_eye_image_test_sequences[i],\
                self.nose_image_test_sequences[i],self.mouth_image_test_sequences[i] = self.load_image_sequence(\
                os.path.join(self.dataset_dir,test_sequences[i]),self.detector,self.predictor)
            self.talking_test[i] = self.get_is_talking(test_sequences[i])
        self.dataset_loaded = True

    def generator(self,batch_size):
        while True:
            indexes = np.arange(len(self.train_sequences))
            np.random.shuffle(indexes)
            for i in range(0,len(indexes),batch_size):
                current_indexes = indexes[i:i+batch_size]

                current_sequences = self.train_sequences[current_indexes]
               
                y = np.zeros((len(current_sequences),))
                for j in range(len(current_sequences)):
                    faces,left_eyes,right_eyes,noses,mouths = self.load_image_sequence(os.path.join(\
                        self.dataset_dir,current_sequences[j]),self.detector,self.predictor)
                    y[j] = self.get_is_talking(current_sequences[j])
                y = y.astype(np.uint8)
                y = np.eye(2)[y]
                
                faces = faces.astype(np.float32)/255
                left_eyes = left_eyes.astype(np.float32)/255
                right_eyes = right_eyes.astype(np.float32)/255
                noses = noses.astype(np.float32)/255
                mouths = mouths.astype(np.float32)/255


                faces = faces.reshape(batch_size,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2])
                left_eyes = left_eyes.reshape(batch_size,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2])
                right_eyes = right_eyes.reshape(batch_size,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2])
                noses = noses.reshape(batch_size,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2])
                mouths = noses.reshape(batch_size,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2])
                yield [faces,left_eyes,right_eyes,noses,mouths],y