from keras.layers import Dropout
class Network(object):
    def __init__(self,input_shape):
        self.input_shape = input_shape
        self.model = self.build()
    def build(self):
        pass
    def train(self):
        pass    