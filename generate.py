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
    checkpoint_path = "./models_checkpoints/betaVAE_latent_8_beta_20.h5"

    model = VanillaVAE(input_shape=input_shape,
                         hidden_dims=[64,128],
                         latent_dim=8,
                         beta = 20)
    # Optimizer 
    model.compile(optimizer=tf.keras.optimizers.Adam(), run_eagerly = True)
    X = np.random.normal(size=(1,28,28,1))
    _ = model(X)
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