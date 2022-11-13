import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from models.VanillaVAE import * 
from models.VQVAE import * 
sns.set(rc={'figure.figsize':(11.7,8.27)})


def load_model(path_to_weights, model_name = 'vanillaVAE') :
    pass 

def store_sampled_images(samples, path_to_save = '') : 
  digit_size = 28
  n = 10
  samples = samples.squeeze()
  figure = np.zeros((digit_size * n, digit_size * n))
  grid_x = np.linspace(-2, 2, n)
  grid_y = np.linspace(-2, 2, n)[::-1]
  for i, _ in enumerate(grid_y):
    for j, _ in enumerate(grid_x): 
      sample = samples[i * n + j]
      figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size,] = sample

  plt.figure(figsize=(15, 15))
  start_range = digit_size // 2
  end_range = n * digit_size + start_range
  pixel_range = np.arange(start_range, end_range, digit_size)
  sample_range_x = np.round(grid_x, 1)
  sample_range_y = np.round(grid_y, 1)
  plt.axis("off")
  #plt.imshow(figure, cmap="Greys_r")
  plt.imsave(path_to_save,figure, cmap="Greys_r")
  plt.cla()



def plot_latent_space(model, X, Y, path = ''):
    points, _ = model.encode(X)
    arr = np.concatenate((points.numpy(),Y),axis = 1)
    df = pd.DataFrame(arr , columns = ['latent_dim_0', 'latent_dim_1', 'y']) 
    pl = sns.scatterplot(
        x="latent_dim_0", y="latent_dim_1",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3
    )
    pl.get_figure().savefig(path)

