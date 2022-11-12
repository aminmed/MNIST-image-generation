import tensorflow as tf
import tensorflow_probability as tfp 
from tensorflow import keras as keras
from keras import layers 
import numpy as np

"""
This code is based on the official tutorial from keras 
https://keras.io/examples/generative/vq_vae/
"""
class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)

class PixelResidualBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(PixelResidualBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])


class PixelCNN(keras.Model):
    
    def __init__(self, 
                 input_shape: int,
                 num_residual_blocks : int,
                 num_pixelcnn_blocks : int,  
                 filters : int,
                 num_embeddings : int) -> None:
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.pixel_input_shape = input_shape
        self.inputs = keras.Input(shape = input_shape)
        
        self.pixelConv_A = PixelConvLayer(
                                          mask_type='A',
                                          filters = filters,
                                          kernel_size = 7,
                                          activation = 'relu', 
                                          padding = 'same'
                                        ) 
    
        modules = []
        for i in range(num_residual_blocks):
            modules.append(
                PixelResidualBlock(filters=filters)
            )

        for i in range(num_pixelcnn_blocks):
            modules.append(
                PixelConvLayer(
                                mask_type='B',
                                filters = filters,
                                kernel_size = 1,
                                strides = 1, 
                                activation = 'relu', 
                                padding = 'valid'
                            ) 
            )
        
        modules.append(
            layers.Conv2D(filters = num_embeddings, kernel_size=1, strides=1, padding='same')
        )

        self.modules = tf.keras.Sequential(layers = modules, name = 'core_module')
        

    def call(self, inputs : tf.Tensor, **kwargs):
        
        x = inputs
        x = tf.one_hot(x, self.num_embeddings)
        x = self.pixelConv_A(x)

        x = self.modules(x)

        return x



class SamplerPixel(keras.Model):

    def __init__(self, pixel_cnn_model) -> None:
        super(SamplerPixel , self).__init__()

        self.pixel_cnn_model = pixel_cnn_model
        self.categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)

    
    def call(self, inputs : tf.Tensor, **kwargs):
        
        x = self.pixel_cnn_model(inputs, training = False)
        x =self.categorical_layer(x)

        return x 
    

    

    
    

        