from copy import deepcopy
import tensorflow as tf
from tensorflow import keras as keras 
from keras import layers 
from typing import *
from models.base import *

class VectorQuantizer(tf.keras.Layer): 
    """
    Reference : https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self, embedding_dim : int ,
                 num_embeddings: int,
                 commitment_cost: float) -> None:
        
        super(VectorQuantizer,self).__init__(name = "vector_quantizer")

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        embedding_shape = [embedding_dim, num_embeddings]
        initializer = tf.keras.initializers.VarianceScaling(distribution='uniform')
        self.embeddings = tf.Variable(initializer(embedding_shape, tf.float32), name='embeddings')


    def __call__(self, z_latents) -> Any:
        
        flattened_z_latents = tf.reshape(z_latents, [-1, self.embedding_dim])

        distances = ( # (z(x) - embeddings)^2 
                tf.reduce_sum(flattened_z_latents**2, 1, keepdims=True)\
                -2 * tf.matmul(flattened_z_latents, self.embeddings)\
                +tf.reduce_sum(self.embeddings**2, 0, keepdims=True)
            )  
        
        encoding_indices = tf.argmax(-distances, 1)

        encodings = tf.one_hot(encoding_indices,self.num_embeddings,dtype=distances.dtype)



    