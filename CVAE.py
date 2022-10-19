import tensorflow as tf
import numpy as np 
from tensorflow.keras import layers


class Sampler(layers.layer) : 
    """
    a layer class to sample z from z_mean and z_log_var
    """
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, inputs):

        z_mean, z_log_var = inputs 
        # get dimensions of the batch and the latent vector representation
        batch_size = tf.shape(z_mean)[0]
        latent_dimension = tf.shape(z_mean)[1]
        # genrerate a random array from N(0,1)
        eps = tf.keras.backend.random_normal(shape = (batch_size, latent_dimension))

        # compute a sampling of z 
        z = z_mean + tf.exp(0.5 * z_log_var) * eps

        return z 

class Encoder(tf.keras.Model): 
    def __init__(self, input_shape,latent_dim,**kwargs) -> None:
        super().__init__()


class Decoder(tf.keras.Model):
    pass

