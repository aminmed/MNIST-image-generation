import tensorflow as tf
from tensorflow import keras as keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
from models.VanillaVAE import *
from models.VQVAE import * 



def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label 


def main(): 
    
    
  batch_size = 128 
  latent_dim = 16
  models = ['vanilla_VAE', 'beta_VAE', 'VQ_VAE','VAE_GAN']
  # loading train and test datasets 
  (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
  X = np.concatenate([x_train, x_test], axis=0)
  X = np.expand_dims(X, -1).astype("float32") / 255

  vae = VQVAE(input_shape=(28,28,1),
                hidden_dims=[32,64],
                latent_dim=latent_dim,
                num_embeddings=128,
                commitement_cost=0.25)
    #vae = VanillaVAE(input_shape=(28,28,1), hidden_dims=[32,64], latent_dim=8)

  vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)

  vae.fit(X, epochs=30, batch_size=batch_size)

    
  vae.save("./save_trained_models/VQVAE")




if __name__ == '__main__' : 
  main()