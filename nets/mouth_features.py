from keras.layers import Dropout
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Input,Concatenate
from keras.layers import LSTM,TimeDistributed,Add,Bidirectional
from keras.models import Model,Sequential
import keras
import numpy as np

class MouthFeatureOnlyNet(object):
    def __init__(self,dataset,input_shape,max_sequence_length):
        self.dataset = dataset
        self.input_shape = input_shape
        self.max_sequence_length = max_sequence_length
        self.model = self.build()
        self.model.summary()
    def build(self):
        mouth_image_model = Sequential()
        mouth_image_model.add(TimeDistributed(Conv2D(32,(3,3),padding='same',activation="relu",strides=(1, 1)),\
                    name="mouth_image_layer1",input_shape=(self.max_sequence_length, self.input_shape[0], self.input_shape[1], self.input_shape[2])))
        
        mouth_image_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        mouth_image_model.add(TimeDistributed(Conv2D(64,kernel_size=(3,3),strides=(1, 1),padding='same',\
                activation="relu",name="mouth_image_layer2")))
        mouth_image_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        mouth_image_model.add(TimeDistributed(Conv2D(128,kernel_size=(3,3),strides=(1, 1),padding='same',\
            activation="relu",name="mouth_image_layer3")))
        mouth_image_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        mouth_image_model.add(TimeDistributed(Flatten()))
        
        mouth_image_model.add(Bidirectional(LSTM(32,return_sequences=True)))
        mouth_image_model.add(Bidirectional(LSTM(128,return_sequences=False)))
        mouth_image_model.add(Dense(256,activation="relu"))

        face_image_model = Sequential()
        face_image_model.add(TimeDistributed(Conv2D(32,(3,3),padding='same',activation="relu",strides=(1, 1)),\
                    name="face_image_layer1",input_shape=(self.max_sequence_length, self.input_shape[0], self.input_shape[1], self.input_shape[2])))
        
        face_image_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        face_image_model.add(TimeDistributed(Conv2D(64,kernel_size=(3,3),strides=(1, 1),padding='same',\
                activation="relu",name="face_image_layer2")))
        face_image_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        face_image_model.add(TimeDistributed(Conv2D(128,kernel_size=(3,3),strides=(1, 1),padding='same',\
            activation="relu",name="face_image_layer3")))
        face_image_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        face_image_model.add(TimeDistributed(Flatten()))
        
        face_image_model.add(Bidirectional(LSTM(32,return_sequences=True)))
        face_image_model.add(Bidirectional(LSTM(128,return_sequences=False)))
        face_image_model.add(Dense(256,activation="relu"))

        merged = keras.layers.concatenate([mouth_image_model.output, face_image_model.output])

        
        merged = Dense(256,activation="relu")(merged)
        merged = Dense(1024,activation="relu")(merged)

        merged = Dense(2,activation="softmax")(merged)

        model = Model(inputs=[mouth_image_model.input,face_image_model.input],outputs=merged)


        return model
        



    def train(self):
        # X_test= [self.dataset.mouth_image_test_sequence, self.dataset.key_points_test_sequence, \
        #     self.dataset.distances_test_sequence, self.dataset.angles_test_sequence]
        X_test= [self.dataset.mouth_image_test_sequence,\
                self.dataset.face_image_test_sequence]

        y_test = self.dataset.Y_test.astype(np.uint8)
        y_test = np.eye(2)[y_test]

        self.model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adam(1e-5),metrics=["accuracy"])
        self.model.fit_generator(self.dataset.generator(1),steps_per_epoch=5000,epochs=50,verbose=1,validation_data=(X_test,y_test))
        
        model_name = "model-mouth-100"
        self.model.save_weights("models/"+model_name+".h5")
        model_json = self.model.to_json()
        with open("models/"+model_name+".json","w+") as json_file:
            json_file.write(model_json)
        score = self.model.evaluate(X_test,y_test)
        with open("logs/log-mouth.txt","a+") as log_file:
            log_file.write("Score of "+model_name+": "+str(score))
            log_file.write("\n")
