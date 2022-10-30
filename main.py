import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
from models.VanillaVAE import *



def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label 


def main(): 
    
    
    batch_size = 128 
    latent_dim = 7
    models = ['vanilla_VAE', 'beta_VAE', 'VQ_VAE','VAE_GAN']
    # loading train and test datasets 

    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X = np.concatenate([x_train], axis = 0)
    X = np.expand_dims(X,-1).astype('float32') / 255

    vae = VanillaVAE(input_shape=(28,28,1),hidden_dims=[32,64], latent_dim=latent_dim)
    
    vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)

  
    vae.fit(X, epochs=30, batch_size=batch_size)

    
    vae.save("./save_trained_models/vanillaVAE")




if __name__ == '__main__' : 
  main()