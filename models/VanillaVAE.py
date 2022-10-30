from copy import deepcopy
import tensorflow as tf
from tensorflow import keras as keras 
from keras import layers 
from typing import *
from models.base import *

class VanillaVAE(baseVAE) :
  
  def __init__(self,
               input_shape : Tuple,
               hidden_dims : List,
               latent_dim : int,
               **kwargs ) -> None:

    super(VanillaVAE, self).__init__()

    self.latent_dim = latent_dim

    if input_shape is None : 
      input_shape = (28,28,1)
    
    if hidden_dims is None : 
      hidden_dims = [32, 64]
    
    self.H, self.W,  self.C  = input_shape
    self.hidden_dims = deepcopy(hidden_dims)
    self.total_loss_tracker = tf.keras.metrics.Mean(name = 'total_loss')
    self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name = 'reconstruction_loss')
    self.kl_loss_tracker = tf.keras.metrics.Mean(name = 'kl_loss')
    #encoder network   

    modules = []

    for hidden_dim in hidden_dims : 
      modules.append(
        tf.keras.Sequential(
          layers = [
            layers.Conv2D(filters=hidden_dim, kernel_size=3, strides=2, padding='same'), 
            layers.BatchNormalization(),
            layers.LeakyReLU()
          ],
          name = 'encoder_module_'+ str(hidden_dim)
        )
      )
    
    modules.append(
      tf.keras.Sequential(
          layers = [
            layers.Flatten(), 
            layers.Dense(units=hidden_dims[-1]),
            layers.LeakyReLU()
          ],
          name = 'encoder_module_dense_layer')
    )
    
    
    self.encoder = tf.keras.Sequential(layers = modules, name = 'encoder')  
    self.mean_out = layers.Dense(units=self.latent_dim, activation='relu')
    self.var_out = layers.Dense(units=self.latent_dim, activation='relu')

    last_layer_col = int(self.W/ (2**len(self.hidden_dims)))
    
    #decoder network 
    self.decoder_input = layers.Dense(units=self.hidden_dims[-1]*(last_layer_col**2))

    hidden_dims.reverse()
    modules = []

    for h in range(len(hidden_dims)-1) : 
      modules.append(
        tf.keras.Sequential(
          layers = [
            layers.Conv2DTranspose(filters=hidden_dims[h], kernel_size=3, strides=2, padding='same'), 
            layers.BatchNormalization(),
            layers.LeakyReLU()
          ],
          name = 'decoder_module_'+ str(hidden_dim)
        )
      )

    modules.append(
      tf.keras.Sequential(
        layers = [
          layers.Conv2DTranspose(filters=hidden_dims[-1], kernel_size=3, strides=2, padding='same'), 
          layers.BatchNormalization(),
          layers.LeakyReLU(),
          layers.Conv2DTranspose(filters=input_shape[2], kernel_size=3, strides=1,padding="same", activation="sigmoid")
        ]
      )
    )   

    self.decoder = tf.keras.Sequential(layers = modules, name = 'decoder')  

  def encode(self, input : tf.Tensor)-> List[tf.Tensor]:
      
    x = self.encoder(input)
    mu , var = self.mean_out(x), self.var_out(x)
    return (mu, var)



  def decode(self , z : tf.Tensor) -> Any: 
    
    w = int(self.W/ (2**len(self.hidden_dims)))

    x = self.decoder_input(z)
    # reshape the decoder input to match the shape of input to conv2transpose
    
    x = layers.Reshape((w,w, self.hidden_dims[-1]))(x)

    x = self.decoder(x)

    return x



  def reparametrize(self, mean: tf.Tensor, log_var : tf.Tensor): 
    
    z_mean, z_log_var = mean, log_var 
    # get dimensions of the batch and the latent vector representation
    batch_size = tf.shape(z_mean)[0]
    latent_dimension = tf.shape(z_mean)[1]
    # genrerate a random array from N(0,1)
    eps = tf.keras.backend.random_normal(shape = (batch_size, latent_dimension))
    # compute a sampling of z using the reparametrization trick 
    z = z_mean + tf.exp(0.5 * z_log_var) * eps

    return z

  def call(self, input : tf.Tensor, **kwargs): 
    
    z_mean, z_log_var = self.encode(input=input)
    z = self.reparametrize(z_mean, z_log_var)

    return (self.decode(z), z_mean, z_log_var)

  def loss_function(self, inputs: Any, **kwargs):
      
      reconstruction, z_mean, z_log_var = self(inputs)

      reconstruction_loss = tf.reduce_mean(
          tf.reduce_sum( tf.keras.losses.binary_crossentropy(inputs, reconstruction),axis = (1,2))
      )
      kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)) 

      total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)

      return  total_loss, reconstruction_loss, kl_loss
  
  def sample(self, num_samples: int, **kwargs):
     
    z = tf.keras.backend.random_normal(shape = (num_samples, self.latent_dim))
    samples = self.decode(z)
    return samples.numpy()

  def generate(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
    return self.call(x)[0]

  @property
  def metrics(self) : 
    return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]
  
  def train_step(self, data): 
    with tf.GradientTape() as tape : 
      
      total_loss, reconstruction_loss, kl_loss = self.loss_function(data)
      
      grads = tape.gradient(total_loss, self.trainable_weights)
      
      self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
      self.total_loss_tracker.update_state(total_loss)
      self.reconstruction_loss_tracker.update_state(reconstruction_loss)
      self.kl_loss_tracker.update_state(kl_loss)

      return {'total_loss' : self.total_loss_tracker.result(),\
              'reconstruction_loss': self.reconstruction_loss_tracker.result(),\
              'kl_loss': self.kl_loss_tracker.result()} 
