import tensorflow as tf
from tensorflow import keras as keras
from keras import layers
from abc import abstractmethod
from typing import *

class baseVAE(tf.keras.Model ) : 
    
    def __init__(self) -> None:
        super(baseVAE, self).__init__()
    
    def encode(self, input : tf.Tensor) -> List[tf.Tensor] : 
        raise NotImplementedError
    
    def decode(self , z : tf.Tensor) -> Any: 
        raise NotImplementedError
    
    def sample(self, num_samples:int, **kwargs):
        raise NotImplementedError
    
    def generate(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def call(self, inputs):
        pass

    @abstractmethod
    def loss_function(self, inputs: Any, **kwargs):
        pass




class ResidualBlock(layers.Layer) : 

    def __init__(self,filters, name = '', **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.resblock=tf.keras.Sequential(
            layers = [
                layers.Conv2D(filters=filters, 
                              kernel_size=3,padding='same',
                              use_bias=False), 
                layers.ReLU(True),
                layers.Conv2D(filters=filters, 
                              kernel_size=3,
                              padding='same',
                              use_bias=False)

            ],
            name = name
        )

    def call(self, inputs, **kwargs):
        return inputs + self.resblock(inputs)

