from typing import *
import tensorflow as tf
from tensorflow import keras as keras 
from keras import layers 


class VectorQuantizer(layers.Layer): 
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
        embedding_shape = (embedding_dim, num_embeddings) 
        initializer = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value = initializer(shape= embedding_shape, dtype = tf.float32), 
            trainable=True,
            name="embeddings")


    def __call__(self, z_latents) -> Any:
        
        z_latents_shape = tf.shape(z_latents)

        flattened_z_latents = tf.reshape(z_latents, [-1, self.embedding_dim]) # [Batch_size * ]

        embedding_sqr = tf.reduce_sum(self.embeddings**2, 0, keepdims=True)
        inputs_sqr = tf.reduce_sum(flattened_z_latents**2, 1, keepdims=True)
        
        distances = ( # (z(x) - embeddings)^2 
                inputs_sqr- 2 * tf.matmul(flattened_z_latents, self.embeddings)\
                +embedding_sqr
            )  
        
        encoding_indices = tf.argmax(-distances, axis=1)

        encodings_one_hot = tf.one_hot(encoding_indices,
                               self.num_embeddings,
                               dtype=distances.dtype
                               )

        reshaped_embeddings = tf.transpose(self.embeddings, [1, 0])

        quantized_latents = tf.matmul(encodings_one_hot, reshaped_embeddings) 
        
        quantized_latents = tf.reshape(quantized_latents, z_latents_shape)

        commitement_loss  = tf.reduce_mean((tf.stop_gradient(quantized_latents) - z_latents)**2)
        embedding_loss    = tf.reduce_mean((tf.stop_gradient(z_latents) - quantized_latents)**2)
    
        vq_loss = embedding_loss + self.commitment_cost * commitement_loss
        #straight-through estimation
        quantized_latents = z_latents + tf.stop_gradient(quantized_latents - z_latents)
        
        return quantized_latents, vq_loss
    
    def get_codebook_indices(self, flattened_inputs):
        
        embedding_sqr = tf.reduce_sum(self.embeddings**2, 0, keepdims=True)
        inputs_sqr    = tf.reduce_sum(flattened_inputs**2, 1, keepdims=True)
        
        distances = ( # (z(x) - embeddings)^2 
                inputs_sqr-2 * tf.matmul(flattened_inputs, self.embeddings)\
                +embedding_sqr
            )  
        
        encoding_indices = tf.argmax(-distances, axis=1)

        return encoding_indices
    
    def get_quantized(self, priors): 

        priors_onehot = tf.one_hot(priors.astype('int32'), self.embeddings ).numpy()
        quantized = tf.matmul(
            priors_onehot.astype("float32"), self.embeddings, transpose_b=True
        )
        quantized = tf.reshape(quantized, (-1,(tf.shape(self.embeddings)[1:-1])))

        return quantized 


class ResidualBlock(layers.Layer) : 

    def __init__(self,filters, name = '', **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.resblock=tf.keras.Sequential(
            layers = [
                layers.Conv2D(filters=filters, 
                              kernel_size=3,padding='same',
                              use_bias=False), 
                layers.ReLU(True),
                layers.Conv2D(filters=filters, 
                              kernel_size=3,
                              padding='same',
                              use_bias=False)

            ],
            name = name
        )

    def call(self, inputs, **kwargs):
        return inputs + self.resblock(inputs)
