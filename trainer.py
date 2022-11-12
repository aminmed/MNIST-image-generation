import numpy as np
import os
# ignore all tensorflow INFO/WARNINGS 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
from tqdm import tqdm, trange
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from models.VanillaVAE import VanillaVAE
from models.VQVAE import VQVAE

SAMPLES_GENERATED_PER_EPOCH = 10 

def store_sampled_images(samples, path_to_save = './train_samples/') : 
  digit_size = 28
  n = len(samples)
  samples = samples.squeeze()
  figure = np.zeros((digit_size * n, digit_size * n))
  grid_x = np.linspace(-1, 1, n)
  grid_y = np.linspace(-1, 1, n)[::-1]
  for i, _ in enumerate(grid_y):
    for j, _ in enumerate(grid_x): 
      sample = samples[i]
      figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size,] = sample

  plt.figure(figsize=(15, 15))
  start_range = digit_size // 2
  end_range = n * digit_size + start_range
  pixel_range = np.arange(start_range, end_range, digit_size)
  sample_range_x = np.round(grid_x, 1)
  sample_range_y = np.round(grid_y, 1)
  plt.axis("off")
  plt.imsave(path_to_save,figure, cmap="Greys_r")



if __name__ == '__main__':
    print("okey")
    parser = argparse.ArgumentParser(description='Train variational autoencoder')
    parser.add_argument('--model',default='VQVAE',const='VQVAE',nargs='?',
                        choices=['vanillaVAE', 'betaVAE', 'VQVAE'],
                        help='model to train vanillaVAE, betaVAE, or VQVAE (default: %(default)s)')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=128,
                      help="The batch size to use for training.")
    parser.add_argument("--optimizer", type=str, default='adam',
                        help="Optimizer to use for training.")

    parser.add_argument("--lr", type=float, default=3e-4,
                      help="The learning rate to use for training.")
    parser.add_argument("--decay_rate", type=float, default=0.9,
                      help="decay rate value to use for learning rate schedule.")
    parser.add_argument("--decay_steps", type=int, default=1000,
                      help="decay steps to use for learning rate schedule.")

    parser.add_argument("--filters", type=int,nargs='+',
                      help="array of filters to use in the model.")
    parser.add_argument("--latent_dim", type = int, default=16, 
                        help="latent dimension to use for the model")
    parser.add_argument("--beta", type=int, default=128, 
                        help= "beta parameter to compromize between reconstruction and kl divergence loss in betaVAE")           
    parser.add_argument("--num_embeddings", type=int, default=128, 
                        help= "number of discrete vector to use in quantization")
    parser.add_argument("--commitement_cost", type=float, default=0.25,
                      help="commitement of embedding vector to encoder output parameter.")

    args = parser.parse_args()

    # Data Pipeline
    print('Train dataset loading...')
    
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    input_shape = (28,28,1)

    X = np.expand_dims(x_train, -1).astype("float32") / 255.0

    print('Train dataset loaded...')
  
    print('Model Loading...')

    if (args.model == 'vanillaVAE'): 
      model = VanillaVAE(input_shape=input_shape,
                         hidden_dims=args.filters,
                         latent_dim=args.latent_dim,
                         beta=1)

    elif (args.model == 'betaVAE'):
      model = VanillaVAE(input_shape=input_shape,
                         hidden_dims=args.filters,
                         latent_dim=args.latent_dim,
                         beta=args.beta)
    else:
      model = VQVAE(input_shape=input_shape,
                    hidden_dims=args.filters,
                    latent_dim=args.latent_dim,
                    num_embeddings=args.num_embeddings,
                    commitement_cost=args.commitement_cost)
    
    print('Model loaded.')
    print(model)
    # Optimizer 
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                  initial_learning_rate=args.lr,
                  decay_steps=args.decay_steps,
                  decay_rate=args.decay_rate
                )
    if args.optimizer == 'adam' : 
      optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else :
      raise NotImplemented("please choose --otimizer adam")
    
    model.compile(optimizer=optimizer,run_eagerly=True)

    # training 

    for epoch in range(args.epochs):
      print(f"EPOCH {epoch+1} / {args.epochs}")
      model.fit(X, epochs = 1, batch_size= args.batch_size)
      if args.model != 'VQVAE':
        samples = model.sample_linspace(num_samples = SAMPLES_GENERATED_PER_EPOCH)
      else : 
        samples, priors = model.sample(num_samples = SAMPLES_GENERATED_PER_EPOCH)
        
      path_to_save_image = "./train_samples/"+ args.model + "_EPOCH_" + str(epoch)+ ".png"
      store_sampled_images(samples,path_to_save_image)
    
    print("Model trained.")
    # save model 
    print("Save model to ./save_models")
    model.save("./save_models/" + args.model)
    print("Model saved")

        
        




