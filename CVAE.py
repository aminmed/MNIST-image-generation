import tensorflow as tf
import numpy as np 
import tensorflow.keras as keras 
from tensorflow.keras import layers


class Sampler(layers.Layer) : 
    """
    a layer class to sample z from z_mean and z_log_var
    """
    def __init__(self) -> None:
        super(Sampler, self).__init__(name = "sampler")
    
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

class Encoder(keras.Model): 

    def __init__(self, latent_dim,input_shape = (28,28,1), name = 'encoder',**kwargs) -> None:
        super(Encoder, self).__init__(name = name, **kwargs)
        # layers 
        self.enc_input = layers.InputLayer(input_shape)
        self.conv_layer_32 = layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', strides = (2,2), padding = 'same')
        self.conv_layer_64 = layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu', strides = (2,2), padding = 'same')
        self.flattening = layers.Flatten()
        self.dense_projection = layers.Dense(units = 64, activation = 'relu')
        self.z_mean = layers.Dense(units = latent_dim, activation = 'relu')
        self.z_log_var = layers.Dense(units = latent_dim, activation = 'relu')

        self.sampling = Sampler()

    
    def call(self, input): 
        
        x = self.enc_input(input)
        x = self.conv_layer_32(x)
        x = self.conv_layer_64(x)
        x = self.flattening(x)
        x = self.dense_projection(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean,z_log_var))
        
        return z_mean, z_log_var, z

class Decoder(keras.Model):
    """Converts z, the encoded digit vector, back into a readable digit."""
    def __init__(self,latent_dim, original_shape = (28,28,1), name = 'decoder',**kwargs) -> None:
        super(Decoder, self).__init__(name = name, **kwargs)
        # layers 
        input_shape = (latent_dim,)
        dense_units = int(original_shape[0]/4 ) * int(original_shape[1]/4 ) * 64
        self.dec_input = layers.InputLayer(input_shape)
        self.dense_projection = layers.Dense(units = dense_units , activation = 'relu')
        self.reshaping = layers.Reshape((int(original_shape[0]/4 ),int(original_shape[1]/4 ), 64))
        self.convTranspose_64 = layers.Conv2DTranspose(filters = 64, kernel_size = 3,  strides = 2, padding = 'same', activation = 'relu')
        self.convTranspose_32 = layers.Conv2DTranspose(filters = 32, kernel_size = 3,  strides = 2, padding = 'same', activation = 'relu')
        self.dec_output = layers.Conv2DTranspose(filters = 1, kernel_size = 3, activation = 'sigmoid', padding = 'same')

    
    def call(self, input): 
        
        x = self.dec_input(input)
        x = self.dense_projection(x)
        x = self.reshaping(x)
        x = self.convTranspose_64(x)
        x = self.convTranspose_32(x)
        return self.dec_output(x)


class CVAE(tf.keras.Model) : 
  def __init__(self, input_shape, latent_dim, **kwargs): 
    super(CVAE,self).__init__(name = "CVAE", **kwargs)
    
    self.encoder = Encoder(latent_dim=latent_dim,input_shape=input_shape)
    self.decoder = Decoder(latent_dim=latent_dim, original_shape=input_shape)
 
    self.total_loss_tracker = tf.keras.metrics.Mean(name = 'total_loss')
    self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name = 'reconstruction_loss')
    self.kl_loss_tracker = tf.keras.metrics.Mean(name = 'kl_loss')

  @property 
  def metrics(self) : 
    return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]
  
  def train_step(self, data): 
    with tf.GradientTape() as tape : 
      
      z_mean, z_log_var,z = self.encoder(data)

      reconstruction = self.decoder(z)

      reconstruction_loss = tf.reduce_mean(
          tf.reduce_sum( tf.keras.losses.binary_crossentropy(data, reconstruction),axis = (1,2))
      )
      kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)) 

      total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
      grads = tape.gradient(total_loss, self.trainable_weights)
      self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
      self.total_loss_tracker.update_state(total_loss)
      self.reconstruction_loss_tracker.update_state(reconstruction_loss)
      self.kl_loss_tracker.update_state(kl_loss)

      return {'total_loss' : self.total_loss_tracker.result(),\
              'reconstruction_loss': self.reconstruction_loss_tracker.result(),\
              'kl_loss': self.kl_loss_tracker.result()} 
