import os
import dlib
import cv2
import json
import numpy as np
from sklearn.model_selection import train_test_split


class MouthFeatureOnlyDataset(object):

    def __init__(self,dataset_dir,bounding_box_dir,image_shape,max_sequence_length):
        self.dataset_dir = dataset_dir
        self.bounding_box_dir = bounding_box_dir
        self.image_shape = image_shape
        self.dataset_loaded = False
        self.max_sequence_length = max_sequence_length
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    def get_dlib_points(self,image,face):
        shape = self.predictor(image,face)
        dlib_points = np.zeros((68,2))
        for i,part in enumerate(shape.parts()):
            dlib_points[i] = [part.x,part.y]
        return dlib_points
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
    def distance_between(self,v1,v2):
        diff = v2 - v1
        diff_squared = np.square(diff)
        dist_squared = diff_squared.sum(axis=1) 
        dists = np.sqrt(dist_squared)
        return dists

    def angles_between(self,v1,v2):
        dot_prod = (v1 * v2).sum(axis=1)
        v1_norm = np.linalg.norm(v1,axis=1)
        v2_norm = np.linalg.norm(v2,axis=1)
        

        cosine_of_angle = (dot_prod/(v1_norm * v2_norm)).reshape(20,1)

        angles = np.arccos(np.clip(cosine_of_angle,-1,1))
        return angles
    def draw_key_points(self,image,key_points):
        for i in range(key_points.shape[0]):
            image = cv2.circle(image,(int(key_points[i][0]),int(key_points[i][1])),1,(255,0,0))
        return image
    def get_mouth_attributes_from_local_frame(self,image,key_points_20):
        
        current_image_shape = image.shape
        top_left = key_points_20.min(axis=0)
        bottom_right = key_points_20.max(axis=0)

        # bound the coordinate system inside eye image
        bottom_right[0] = min(current_image_shape[1],bottom_right[0]+5)
        bottom_right[1] = min(current_image_shape[0],bottom_right[1]+5)
        top_left[0] = max(0,top_left[0]-5)
        top_left[1] = max(0,top_left[1]-5)

        # crop the eye
        top_left = top_left.astype(int)
        bottom_right = bottom_right.astype(int)
        mouth_image = image[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
        if mouth_image.shape[0]==0:
            # self.draw_key_points(image,key_points_20)  
            # cv2.imshow("Image",image)
            # cv2.waitKey(0)
            # cv2.destoryAllWindows(0)
            image = np.zeros((self.image_shape[0],self.image_shape[1],self.image_shape[2]))
            key_points = np.zeros((20,2))
            dists = np.zeros((20))
            angles = np.zeros((20))
            return image, key_points,dists,angles

        # translate the eye key points from face image frame to eye image frame
        key_points = key_points_20 - top_left
        key_points +=np.finfo(float).eps
        # horizontal scale to resize image
        scale_h = self.image_shape[1]/float(mouth_image.shape[1])
        # vertical scale to resize image
        scale_v = self.image_shape[0]/float(mouth_image.shape[0])

        # resize left eye image to network input size
        mouth_image = cv2.resize(mouth_image,(self.image_shape[0],self.image_shape[1]))

        # scale left key points proportional with respect to left eye image resize scale
        scale = np.array([[scale_h,scale_v]])
        key_points = key_points * scale 

        # calculate centroid of left eye key points 
        centroid = np.array([key_points.mean(axis=0)])

        # calculate distances from  centroid to each left eye key points
        dists = self.distance_between(key_points,centroid)

        # calculate angles between centroid point vector and left eye key points vectors
        angles = self.angles_between(key_points,centroid)
        return mouth_image, key_points,dists,angles

    def get_mouth_features_from_image(self,image,bounding_box):
        face = dlib.rectangle(int(bounding_box[0]),int(bounding_box[1]),int(bounding_box[2]),int(bounding_box[3]))

        # cv2.rectangle(image,(face.left(),face.top()),(face.right(),face.bottom()),(255,255,0))
        b_box_array = np.array(bounding_box)
        key_points = self.get_dlib_points(image,face)
        mouth_key_points = key_points[48:68]
        image = self.draw_key_points(image,mouth_key_points)
        assert len(mouth_key_points) == 20, "Mouth key points should be twenty points"
        mouth_image, key_points,dists,angles = self.get_mouth_attributes_from_local_frame(image,mouth_key_points)
        return mouth_image, key_points,dists,angles
    def get_mouth_features(self,sequence_path):
        bboxes = self.get_bounding_boxes(sequence_path)
        output_images = np.zeros((self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        output_faces = np.zeros((self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        output_key_points = np.zeros((self.max_sequence_length,20,2))
        output_distances = np.zeros((self.max_sequence_length,20))
        output_angles = np.zeros((self.max_sequence_length,20))
        img_files = os.listdir(sequence_path)
        img_files.sort()

        for i in range(len(img_files)):
            img = cv2.imread(os.path.join(sequence_path,img_files[i]))
            bounding_box = bboxes[img_files[i]]
            if not(img is None):
                
                face_image = img[
                    max(0,int(bounding_box[1]-5)):min(img.shape[1],int(bounding_box[3])),
                    max(0,int(bounding_box[0]-5)):min(img.shape[0],int(bounding_box[2]))
                ]
                try:
                    face_image = cv2.resize(face_image,(self.image_shape[0],self.image_shape[1]))
                except:
                    # print bounding_box
                    # cv2.imshow("Image",img)
                    # # cv2.imshow("Face image",face_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    face_image = np.zeros((self.image_shape[0],self.image_shape[1],self.image_shape[2]))
                mouth_image,kps,dists,angles = self.get_mouth_features_from_image(img,bounding_box)
                # self.draw_key_points(mouth_image,kps)
                # cv2.imshow("Image",img)
                # cv2.imshow("Mouth Image",mouth_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                output_faces[i] = face_image
                output_images[i] = mouth_image
                output_key_points[i] = kps
                output_distances[i] = dists
                output_angles[i] = angles.reshape((20,))
            else:
                raise Exception("Unable to read image form "+os.path.join(sequence_path,img_files[i]))
        return output_faces, output_images,output_key_points,output_distances,output_angles
    def get_is_talking(self,folder_name):
        if folder_name.lower().count("talking")>0:
            return 1
        else:
            return 0
    def load_dataset(self):
        sequences = os.listdir(self.dataset_dir)
        # sequences = sequences[:3000]
        train_sequences,test_sequences = train_test_split(sequences,test_size=0.1)
        num_train_sequences  = len(train_sequences)
        num_test_sequences  = len(test_sequences)
        
        self.face_image_train_sequence = np.zeros((num_train_sequences,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        self.mouth_image_train_sequence = np.zeros((num_train_sequences,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        self.key_points_train_sequence = np.zeros((num_train_sequences,self.max_sequence_length,20,2))
        self.distances_train_sequence = np.zeros((num_train_sequences,self.max_sequence_length,20))
        self.angles_train_sequence = np.zeros((num_train_sequences,self.max_sequence_length,20))
        self.Y_train = np.zeros((num_train_sequences,),dtype=np.uint8)


        
        
        
        self.face_image_test_sequence = np.zeros((num_test_sequences,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        self.mouth_image_test_sequence = np.zeros((num_test_sequences,self.max_sequence_length,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
        self.key_points_test_sequence = np.zeros((num_test_sequences,self.max_sequence_length,20,2))
        self.distances_test_sequence = np.zeros((num_test_sequences,self.max_sequence_length,20))
        self.angles_test_sequence = np.zeros((num_test_sequences,self.max_sequence_length,20))
        self.Y_test = np.zeros((num_test_sequences,),dtype=np.uint8)

        print "Loading",num_train_sequences,"train sequences"
        for i in range(num_train_sequences):
            faces,mouths,points,distances, angles = self.get_mouth_features(os.path.join(self.dataset_dir, train_sequences[i]))
            self.face_image_train_sequence[i] = faces
            self.mouth_image_train_sequence[i] = mouths
            self.key_points_train_sequence[i] = points
            self.distances_train_sequence[i] = distances
            self.angles_train_sequence[i] = angles
            self.Y_train[i] = self.get_is_talking(train_sequences[i])
            if (i+1)%100==0:
                print "loaded",i+1,"sequences"
        print "Loaded",num_train_sequences,"train sequences"

        print "Loading test sequences"
        for i in range(num_test_sequences):
            faces,images,points,distances, angles = self.get_mouth_features(os.path.join(self.dataset_dir,test_sequences[i]))
            self.face_image_test_sequence[i] = faces
            self.mouth_image_test_sequence[i] = images
            self.key_points_test_sequence[i] = points
            self.distances_test_sequence[i] = distances
            self.angles_test_sequence[i] = angles
            self.Y_test[i] = self.get_is_talking(test_sequences[i])
            if (i+1)%100==0:
                print "loaded",i+1,"sequences"
            
        print "Loaded test sequences"


        print "Preprocessing dataset"
        # Normalize images

        self.face_image_train_sequence = self.face_image_train_sequence.astype(np.float32)/255.0
        self.face_image_test_sequence = self.face_image_test_sequence.astype(np.float32)/255.0
        # Normalize images

        self.mouth_image_train_sequence = self.mouth_image_train_sequence.astype(np.float32)/255.0
        self.mouth_image_test_sequence = self.mouth_image_test_sequence.astype(np.float32)/255.0

        # Normalize key points 
        image_width = self.image_shape[0]
        self.key_points_train_sequence = self.key_points_train_sequence.astype(np.float32)/float(image_width)
        self.key_points_test_sequence = self.key_points_test_sequence.astype(np.float32)/float(image_width)

        # Expand dims for network input
        self.key_points_train_sequence = np.expand_dims(self.key_points_train_sequence,2)
        self.key_points_test_sequence = np.expand_dims(self.key_points_test_sequence,2)

        # Normalize distances 
        self.distances_train_sequence = self.distances_train_sequence.astype(np.float32)/float(image_width)
        self.distances_test_sequence = self.distances_test_sequence.astype(np.float32)/float(image_width)
        
        # Expand dims for network input
        self.distances_train_sequence = np.expand_dims(self.distances_train_sequence,2)
        self.distances_train_sequence = np.expand_dims(self.distances_train_sequence,4)

        self.distances_test_sequence = np.expand_dims(self.distances_test_sequence,2)
        self.distances_test_sequence = np.expand_dims(self.distances_test_sequence,4)

        # Normalize angles 
        self.angles_train_sequence = self.angles_train_sequence.astype(np.float32)/np.pi
        self.angles_test_sequence = self.angles_test_sequence.astype(np.float32)/np.pi
         # Expand dims for network input
        self.angles_train_sequence = np.expand_dims(self.angles_train_sequence,2)
        self.angles_train_sequence = np.expand_dims(self.angles_train_sequence,4)

        self.angles_test_sequence = np.expand_dims(self.angles_test_sequence,2)
        self.angles_test_sequence = np.expand_dims(self.angles_test_sequence,4)
        print "All datasets are loaded and preprocessed"
        self.dataset_loaded = True
    def generator(self,batch_size):
        while True:
            indexes = range(len(self.mouth_image_train_sequence))
            np.random.shuffle(indexes)
            for i in range(0,len(indexes),batch_size):
                current_indexes = indexes[i:i+batch_size]
                f_images = self.face_image_train_sequence[current_indexes]
                m_images = self.mouth_image_train_sequence[current_indexes]
                # kpoints = self.key_points_train_sequence[current_indexes]
                # dpoints = self.distances_train_sequence[current_indexes]
                # angles = self.angles_train_sequence[current_indexes]

                y = self.Y_train[current_indexes]
                y = np.eye(2)[y]
                # yield  [m_images,kpoints,dpoints,angles],y
                yield  [m_images, f_images],y