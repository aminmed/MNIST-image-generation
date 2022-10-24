import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt 

from CVAE import *



def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label 


def main(): 
    
    
    batch_size = 128 
    latent_dim = 7
    models = ['vanilla_VAE', 'beta_VAE', 'VQ_VAE','VAE_GAN']
    # loading train and test datasets 

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    X = np.concatenate([x_train, x_test], axis = 0)
    X = np.expand_dims(X,-1).astype('float32') / 255

    vae = CVAE(input_shape=(28,28,1), latent_dim=latent_dim)
    vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
    vae.fit(X, epochs=20, batch_size=batch_size)
    
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-1, 1, n)
    grid_y = np.linspace(-1, 1, n)[::-1]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size,] = digit
            plt.figure(figsize=(15, 15))
            start_range = digit_size // 2
            end_range = n * digit_size + start_range
            pixel_range = np.arange(start_range, end_range, digit_size)
            sample_range_x = np.round(grid_x, 1)
            sample_range_y = np.round(grid_y, 1)
            plt.xticks(pixel_range, sample_range_x)
            plt.yticks(pixel_range, sample_range_y)
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            plt.axis("off")
            plt.imshow(figure, cmap="Greys_r")





main()