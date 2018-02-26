from keras.layers import Dropout
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Input,concatenate,Concatenate
from keras.layers import LSTM,TimeDistributed,Add
from keras.models import Model,Sequential
import keras
import numpy as np

class Network(object):
    def __init__(self,dataset,input_shape,max_sequence_length):
        self.dataset = dataset
        self.input_shape = input_shape
        self.max_sequence_length = max_sequence_length
        self.model = self.build()
        self.model.summary()
    def build(self):
        
        face_model = Sequential()
        face_model.add(TimeDistributed(Conv2D(32,(3,3),padding='valid',activation="relu",strides=(1, 1)),\
                    name="face_layer1",input_shape=(self.max_sequence_length, self.input_shape[0], self.input_shape[1], self.input_shape[2])))
        
        face_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        face_model.add(TimeDistributed(Conv2D(64,kernel_size=(3,3),strides=(1, 1),padding='valid',\
                activation="relu",name="face_layer2")))
        face_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        face_model.add(TimeDistributed(Conv2D(128,kernel_size=(3,3),strides=(1, 1),padding='valid',\
            activation="relu",name="face_layer3")))
        face_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        face_model.add(TimeDistributed(Flatten()))
        
        nose_model = Sequential()
        nose_model.add(TimeDistributed(Conv2D(32,(3,3),padding='valid',activation="relu",strides=(1, 1)),\
                    name="nose_layer1",input_shape=(self.max_sequence_length, self.input_shape[0], self.input_shape[1], self.input_shape[2])))
        
        nose_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        nose_model.add(TimeDistributed(Conv2D(64,kernel_size=(3,3),strides=(1, 1),padding='valid',\
                activation="relu",name="nose_layer2")))
        nose_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        nose_model.add(TimeDistributed(Conv2D(128,kernel_size=(3,3),strides=(1, 1),padding='valid',\
            activation="relu",name="nose_layer3")))
        nose_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        nose_model.add(TimeDistributed(Flatten()))
        
        left_eye_model = Sequential()
        left_eye_model.add(TimeDistributed(Conv2D(32,(3,3),padding='valid',activation="relu",strides=(1, 1)),\
                    name="left_eye_layer1",input_shape=(self.max_sequence_length, self.input_shape[0], self.input_shape[1], self.input_shape[2])))
    
        left_eye_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        left_eye_model.add(TimeDistributed(Conv2D(64,kernel_size=(3,3),strides=(1, 1),padding='valid',\
                activation="relu",name="left_eye_layer2")))
        left_eye_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        left_eye_model.add(TimeDistributed(Conv2D(128,kernel_size=(3,3),strides=(1, 1),padding='valid',\
            activation="relu",name="left_eye_layer3")))
        left_eye_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        left_eye_model.add(TimeDistributed(Flatten()))

        right_eye_model = Sequential()
        right_eye_model.add(TimeDistributed(Conv2D(32,(3,3),padding='valid',activation="relu",strides=(1, 1)),\
                    name="right_eye_layer1",input_shape=(self.max_sequence_length, self.input_shape[0], self.input_shape[1], self.input_shape[2])))
        
        right_eye_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        right_eye_model.add(TimeDistributed(Conv2D(64,kernel_size=(3,3),strides=(1, 1),padding='valid',\
                activation="relu",name="right_eye_layer2")))
        right_eye_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        right_eye_model.add(TimeDistributed(Conv2D(128,kernel_size=(3,3),strides=(1, 1),padding='valid',\
            activation="relu",name="right_eye_layer3")))
        right_eye_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        right_eye_model.add(TimeDistributed(Flatten()))

        mouth_model = Sequential()
        mouth_model.add(TimeDistributed(Conv2D(32,(3,3),padding='valid',activation="relu",strides=(1, 1)),\
                    name="mouth_layer1",input_shape=(self.max_sequence_length, self.input_shape[0], self.input_shape[1], self.input_shape[2])))
        
        mouth_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        mouth_model.add(TimeDistributed(Conv2D(64,kernel_size=(3,3),strides=(1, 1),padding='valid',\
                activation="relu",name="mouth_layer2")))
        mouth_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        mouth_model.add(TimeDistributed(Conv2D(128,kernel_size=(3,3),strides=(1, 1),padding='valid',\
            activation="relu",name="mouth_layer3")))
        mouth_model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        mouth_model.add(TimeDistributed(Flatten()))
        merged_layer = Add()([face_model.output,left_eye_model.output,right_eye_model.output,nose_model.output,mouth_model.output])
        
        dense1 = TimeDistributed(Dense(128,activation="relu"))(merged_layer)   
        dropout1 = TimeDistributed(Dropout(0.2))(dense1)
        dense2 = TimeDistributed(Dense(256,activation="relu"))(dropout1)
        dropout2 = TimeDistributed(Dropout(0.2))(dense2)
        lstm1 = LSTM(32,activation='relu',return_sequences=True,stateful=False)(dropout2)
        lstm2 = LSTM(64,activation='relu',return_sequences=False,stateful=False)(lstm1)
        
        dense3 = Dense(256,activation="relu")(lstm2)
        output = Dense(2,activation="softmax")(dense3)

        # model = Model(inputs=[face_layer,left_eye_layer_input,right_eye_layer_input,nose_layer_input,mouth_layer_input],outputs=output)
        model = Model(inputs=[face_model.input,left_eye_model.input,right_eye_model.input,nose_model.input,mouth_model.input],\
                            outputs = output
                            )
        return model



    def train(self):
        X_test= [self.dataset.face_image_test_sequences, self.dataset.left_eye_image_test_sequences, \
            self.dataset.right_eye_image_test_sequences, self.dataset.nose_image_test_sequences, \
            self.dataset.mouth_image_test_sequences]
        y_test = self.dataset.talking_test.astype(np.uint8)
        print y_test[0]
        y_test = np.eye(2)[y_test]

        self.model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(1e-4),metrics=["accuracy"])
        self.model.fit_generator(self.dataset.generator(32),steps_per_epoch=300,epochs=10,verbose=1,validation_data=(X_test,y_test))
        self.model.save_weights("models/model.h5")
        model_json = self.model.to_json()
        with open("models/model.json","w+") as json_file:
            json_file.write(model_json)
        score = self.model.evaluate(X_test,y_test)
        with open("logs/log.txt","w+") as log_file:
            log_file.write("Score: "+str(score))
            log_file.write("\n")