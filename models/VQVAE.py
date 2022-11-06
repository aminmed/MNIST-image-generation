from copy import deepcopy
import tensorflow as tf
from tensorflow import keras as keras 
from keras import layers 
from typing import *
from models.base import *

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
        embedding_shape = [embedding_dim, num_embeddings]
        initializer = tf.keras.initializers.VarianceScaling(distribution='uniform')
        self.embeddings = tf.Variable(
            initializer(embedding_shape, tf.float32), 
            trainable=True,
            name="embeddings")


    def __call__(self, z_latents) -> Any:
        
        z_latents_shape = tf.shape(z_latents)

        flattened_z_latents = tf.reshape(z_latents, [-1, self.embedding_dim]) # [Batch_size * ]

        embedding_sqr = tf.reduce_sum(self.embeddings**2, 0, keepdims=True)
        inputs_sqr = tf.reduce_sum(flattened_z_latents**2, 1, keepdims=True)
        
        distances = ( # (z(x) - embeddings)^2 
                inputs_sqr-2 * tf.matmul(flattened_z_latents, self.embeddings)\
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


class ResidualBlock(layers.Layer) : 

    def __init__(self,input_shape,filters, name = '', **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.resblock=tf.keras.Sequential(
            layers = [
                layers.Conv2D(filters=filters, 
                              kernel_size=3,padding='same',
                              input_shape = input_shape,
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

class VQVAE(baseVAE): 
    
    def __init__(self,
                 input_shape : Tuple,
                 hidden_dims : List,
                 latent_dim : int,
                 embeddings_num : int,
                 commitement_cost : float, 
                 beta : float, 
               **kwargs ) -> None:
        super(VQVAE, self).__init__()

        self.beta = beta 
        self.latent_dim = latent_dim   # same as embedding dimension
        self.hidden_dims = deepcopy(hidden_dims)
        self.embeddings_num = embeddings_num
        self.commitement_cost = commitement_cost
        self.H, self.W,  self.C  = input_shape

        self.total_loss_tracker = tf.keras.metrics.Mean(name = 'total_loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name = 'reconstruction_loss')
        self.vq_loss_tracker = tf.keras.metrics.Mean(name = 'vq_loss')

        modules = []

        for hidden_dim in hidden_dims : 
      
          modules.append(
            tf.keras.Sequential(
              layers = [
                layers.Conv2D(filters=hidden_dim, kernel_size=3, strides=2, padding='same'), 
                layers.BatchNormalization(),
                layers.LeakyReLU()
              ],
              name = 'encoder_module_'+ str(hidden_dim)
            )
          )
    
        modules.append(
          tf.keras.Sequential(
              layers = [
                layers.Flatten(), 
                layers.Dense(units=hidden_dims[-1]),
                layers.LeakyReLU()
              ],
              name = 'encoder_module_dense_layer')
        )
    
    
        self.encoder = tf.keras.Sequential(layers = modules, name = 'encoder')  
        self.encoder_output = layers.Dense(units=self.latent_dim, activation='relu')

        self.last_layer_encoder_w = int(self.W/ (2**len(self.hidden_dims)))

        self.vq_layer = VectorQuantizer(embedding_dim=self.latent_dim,
                                        num_embeddings=self.embeddings_num,
                                        commitment_cost=self.commitement_cost)
        


        #decoder network 

        self.decoder_input = layers.Dense(
                            units=self.hidden_dims[-1]*(self. last_layer_encoder_w**2))

        hidden_dims.reverse()
        modules = []

        for h in range(len(hidden_dims)) : 
          modules.append(
            tf.keras.Sequential(
                layers = [
                    layers.Conv2DTranspose(filters=hidden_dims[h], 
                                       kernel_size=3,
                                       strides=2, padding='same'), 
                    layers.BatchNormalization(),
                    layers.LeakyReLU()
                    ],
                name = 'decoder_module_'+ str(hidden_dims[h])
                )
            )


        kernel_out_decoder = (1,5)[len(self.hidden_dims)>2]

        modules.append(
            tf.keras.Sequential(
            layers = [
                layers.Conv2DTranspose(filters=input_shape[2], kernel_size= kernel_out_decoder, strides=1, padding='valid', activation="sigmoid")
            ]
            )
        )   

        self.decoder = tf.keras.Sequential(layers = modules, name = 'decoder')  


    def call(self, inputs):
        
        Z_e = self.encode(inputs)
        Z_q , vq_loss = self.vq_layer(Z_e)

        return self.decode(Z_q), Z_e, vq_loss

    def loss_function(self, inputs: tf.Tensor, **kwargs):
        
        reconstruction , Z_e, vq_loss = self(inputs)
        
        reconstruction_loss = tf.reduce_mean(
          tf.reduce_sum( tf.keras.losses.binary_crossentropy(inputs, reconstruction),
                        axis = (1,2))
            ) 

        total_loss = reconstruction_loss + self.beta * tf.reduce_mean(vq_loss)

        return total_loss, reconstruction_loss, vq_loss

    def encode(self, input: tf.Tensor) -> tf.Tensor:
        
        x = self.encoder(input)
        Z_e = self.encoder_output(x)
        
        return Z_e
    
    def decode(self, Z_q: tf.Tensor) -> Any:
        
        w = self.last_layer_encoder_w

        x = self.decoder_input(Z_q)
        # reshape the decoder input to match the shape of input to conv2transpose 
        x = layers.Reshape((w,w, self.hidden_dims[-1]))(x)

        x = self.decoder(x)

        return x

    @property
    def metrics(self) : 
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.vq_loss_tracker]
  
    def train_step(self, data): 
        with tf.GradientTape() as tape : 
            total_loss, reconstruction_loss, vq_loss = self.loss_function(data)
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.vq_loss_tracker.update_state(vq_loss)

        return {'total_loss' : self.total_loss_tracker.result(),\
              'reconstruction_loss': self.reconstruction_loss_tracker.result(),\
              'vq_loss': self.vq_loss_tracker.result()} 

    def sample(self, num_samples: int, **kwargs):
        pass

    def generate(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        return self(x)[0]