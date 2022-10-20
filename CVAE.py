import tensorflow as tf
import numpy as np 
from tensorflow.keras import layers


class Sampler(layers.layer) : 
    """
    a layer class to sample z from z_mean and z_log_var
    """
    def __init__(self) -> None:
        super(Sampler, self).__init__()
    
    def __call__(self, inputs):

        z_mean, z_log_var = inputs 
        # get dimensions of the batch and the latent vector representation
        batch_size = tf.shape(z_mean)[0]
        latent_dimension = tf.shape(z_mean)[1]
        # genrerate a random array from N(0,1)
        eps = tf.keras.backend.random_normal(shape = (batch_size, latent_dimension))

        # compute a sampling of z using the reparametrization trick 
        z = z_mean + tf.exp(0.5 * z_log_var) * eps

        return z 

class Encoder(layers.Layer): 

    def __init__(self, latent_dim,input_shape = (28,28,1), name = 'encoder',**kwargs) -> None:
        super(Encoder, self).__init__(name, **kwargs)
        # layers 
        self.input = tf.keras.Input(input_shape)
        self.conv_layer_32 = layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', strides = (2,2), padding = 'same')
        self.conv_layer_64 = layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu', strides = (2,2), padding = 'same')
        self.flattening = layers.Flatten()
        self.dense_projection = layers.Dense(units = 64, activation = 'relu')
        self.z_mean = layers.Dense(units = latent_dim, activation = 'relu')
        self.z_log_var = layers.Dense(units = latent_dim, activation = 'relu')

        self.sampling = Sampler()

    
    def call(self, input): 
        
        x = self.input(x)
        x = self.conv_layer_32(x)
        x = self.conv_layer_64(x)
        x = self.flattening(x)
        x = self.dense_projection(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean,z_log_var))
        
        return z_mean, z_log_var, z

class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""
    def __init__(self,latent_dim, original_shape = (28,28,1), name = 'decoder',**kwargs) -> None:
        super(Encoder, self).__init__(name, **kwargs)
        # layers 
        input_shape = (latent_dim,)
        (original_shape[0]/4 ) * (original_shape[1]/4 ) * 64
        self.input = tf.keras.Input(input_shape)
        self.dense_projection = layers.Dense(units = (original_shape[0]/4 ) * (original_shape[1]/4 ) * 64 , activation = 'relu')
        self.reshaping = layers.Reshape(((original_shape[0]/4 ),(original_shape[1]/4 ), 64))
        self.convTranspose_64 = layers.Conv2DTranspose(filters = 64, kernels = 3,  strides = 2, padding = 'same', activation = 'relu')
        self.convTranspose_32 = layers.Conv2DTranspose(filters = 32, kernels = 3,  strides = 2, padding = 'same', activation = 'relu')
        self.output = layers.Conv2DTranspose(filters = 1, kernels = 3, activation = 'sigmoid', padding = 'same')

    
    def call(self, input): 
        
        x = self.input(x)
        x = self.dense_projection(x)
        x = self.reshaping(x)
        x = self.convTranspose_64(x)
        x = self.convTranspose_32(x)
        return self.output(x)



class CVAE(tf.keras.Model):
    pass
