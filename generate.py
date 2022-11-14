import os
import numpy as np  
import argparse
from tqdm import tqdm, trange

import tensorflow as tf
from PIL import Image
from models.VQVAE import *
from models.VanillaVAE import * 



def load_model():
    input_shape = (28,28,1)
    model_type = 'betaVAE'
    checkpoint_path = "./models_checkpoints/betaVAE_latent_16_beta_50.h5"

    if model_type == 'VQVAE' : 
        model = VQVAE(input_shape=input_shape,
                  hidden_dims=[64,128],
                  latent_dim=16,
                  num_embeddings=128,
                  commitement_cost=0.25)

        model.compile(optimizer=tf.keras.optimizers.Adam(), run_eagerly = True)
        x_ = np.random.normal(size=(1,28,28,1))
        _ = model(x_)
        model.load_weights(checkpoint_path)
        model.load_weights_for_pixel_cnn("./models_checkpoints/pixel_VQVAE_16_128.h5")



    else : 
        model = VanillaVAE(input_shape=input_shape,
                  hidden_dims=[64,128],
                  latent_dim=16,
                  beta = 1)

        model.compile(optimizer=tf.keras.optimizers.Adam(), run_eagerly = True)
        x_ = np.random.normal(size=(1,28,28,1))
        _ = model(x_)
        model.load_weights(checkpoint_path)

    
    return model


if __name__ == '__main__':

    # ignore all tensorflow INFO/WARNINGS 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    parser = argparse.ArgumentParser(description='Variational AutoEncoder .')
    parser.add_argument("--n_samples", type=int, default=1024,
                      help="Number of generated samples.")

    args = parser.parse_args()

    print('Model Loading...')
    model = load_model()
    print("Model Loaded ...")

    print('Start Generating :')
    
    os.makedirs('./samples', exist_ok=True)
   
    with trange(args.n_samples, desc="Generated", unit="img") as te:
        for idx in te:
            x = model.sample(num_samples=1)
            x = x.reshape((28,28,1))
            x_img = tf.keras.utils.array_to_img(x)
            x_img.save(os.path.join('./samples', f'{idx}.png'))