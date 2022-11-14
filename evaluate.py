import numpy as np
import os
import math
# ignore all tensorflow INFO/WARNINGS 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from models.VanillaVAE import VanillaVAE
from models.VQVAE import VQVAE

from metrics.IS_score import *
from metrics.frechet_kernel_inception_distance import * 

def load_samples_from_repository(path) : 
    samples = np.array()
    # impelementation !! 
    return samples 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate variational autoencoder')
    parser.add_argument('--model',default='VQVAE',const='VQVAE',nargs='?',
                        choices=['vanillaVAE', 'betaVAE', 'VQVAE'],
                        help='model to evaluate : vanillaVAE, betaVAE, or VQVAE (default: %(default)s)')
    parser.add_argument('--metric',default='IS',const='IS',nargs='?',
                        choices=['IS', 'FID', 'CAS'],
                        help='metric of evaluation FID, IS or CAS (default: %(default)s)')

    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of samples to generate.")

    parser.add_argument("--path", type=str, default="./evaluation",
                      help="The path to the folder to store evaluation results.")
    parser.add_argument("--path_samples", type=str, default="None",
                      help="The path to the folder where samples are stored.")

    args = parser.parse_args()

    if args.path_samples != None: 
        samples = load_samples_from_repository(args.path_samples)

    else : 
        model = tf.keras.models.load_model("./save_models/" + args.model ) 
        samples = model.sample(args.num_samples)
    
    if args.metric == "IS" : 
        
        samples = samples.reshape((-1,28,28,1))
        is_score = get_inception_score(samples, splits= 10)
        print(f"Inception score for given model is {is_score}")
    elif args.metric == "FID":
        pass
    
    else : 
        raise NotImplemented("CAS is not implemented yet !")

    
    


    


