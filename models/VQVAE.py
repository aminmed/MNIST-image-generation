from copy import deepcopy
from typing import *
import tensorflow as tf
from tensorflow import keras as keras 
from keras import layers 
from models.base import *
from models.VectorQuantizers import *
from models.PixelCNN import *  

class VQVAE(baseVAE): 
    
    def __init__(self,
                 input_shape : Tuple,
                 hidden_dims : List,
                 latent_dim : int,
                 num_embeddings : int,
                 commitement_cost : float, 
               **kwargs ) -> None:
        super(VQVAE, self).__init__()

        self.latent_dim = latent_dim   # same as embedding dimension
        self.hidden_dims = deepcopy(hidden_dims)
        self.num_embeddings = num_embeddings
        self.commitement_cost = commitement_cost
        self.H, self.W,  self.C  = input_shape

        # pixel cnn model and sampler 

        self.pixel_cnn = None
        self.sampler = None 

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
        
        #for _ in range(3): 
        #    modules.append(
        #        ResidualBlock(filters= 64, name = 'encoder_resblock' + str(_))
        #    )

    
    
        self.encoder = tf.keras.Sequential(layers = modules, name = 'encoder')  
        
        self.encoder_output = layers.Conv2D(filters = self.latent_dim,
                                            kernel_size = 1,
                                            padding='same')



        self.vq_layer = VectorQuantizer(embedding_dim=self.latent_dim,
                                        num_embeddings=self.num_embeddings,
                                        commitment_cost=self.commitement_cost)
        


        #decoder network 


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


        modules.append(
            tf.keras.Sequential(
            layers = [
                layers.Conv2DTranspose(filters=input_shape[2], kernel_size= 3, strides=1, padding='same', activation="sigmoid")
            ]
            )
        )   

        modules.append(
            layers.Resizing(
                height=input_shape[1], width = input_shape[2], interpolation="bilinear", crop_to_aspect_ratio=False
                )
        )

        self.decoder = tf.keras.Sequential(layers = modules, name = 'decoder')  


    def call(self, inputs):
        
        Z_e = self.encode(inputs)
        Z_q , vq_loss = self.vq_layer(Z_e)

        return self.decode(Z_q), Z_e, vq_loss

    def loss_function(self, inputs: tf.Tensor, **kwargs):
        
        reconstruction , Z_e, vq_loss = self(inputs)
        
        reconstruction_loss = tf.reduce_mean((inputs - reconstruction) ** 2) 

        total_loss = reconstruction_loss + tf.reduce_sum(vq_loss)

        return total_loss, reconstruction_loss, vq_loss

    def encode(self, inputs: tf.Tensor) -> tf.Tensor:
        
        x = self.encoder(inputs)
        Z_e = self.encoder_output(x)
        
        return Z_e
    
    def decode(self, Z_q: tf.Tensor) -> Any:

        x = self.decoder(x)

        return x
    
    def train_pixel_cnn(self, X_train : tf.Tensor,
                        epochs = 30,
                        num_residual_blocks=2,
                        num_pixelcnn_blocks = 2,
                        filters = 128):
        
        pixel_input_shape = (7,7) # not well generalized , should be written using encoder output shape 
        self.pixel_cnn = PixelCNN(input_shape=pixel_input_shape,
                                  num_residual_blocks=num_residual_blocks, 
                                  num_pixelcnn_blocks=num_pixelcnn_blocks,
                                  filters= filters, 
                                  num_embeddings=self.num_embeddings)
        
        # preprocess data to train pixelcnn 
        print("##### train_PixelCNN : data preparation ")
        encoded_data = self.encode(X_train).numpy()
        flattened_data = encoded_data.reshape(-1, encoded_data.shape[-1])
        codebook_indices = self.vq_layer.get_codebook_indices(flattened_data)
        codebook_indices = codebook_indices.numpy().reshape((-1,pixel_input_shape))
        print("##### train_PixelCNN : training")
        self.pixel_cnn.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        
        self.pixel_cnn.fit(
            x=codebook_indices,
            y=codebook_indices,
            batch_size=128,
            epochs=epochs,
            validation_split=0.1,
        )
        print("##### train_PixelCNN : training completed successfully sampler creation ")

        self.sampler = SamplerPixel(pixel_cnn_model = self.pixel_cnn)

        return True

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
        
        if self.sampler is None : 
            raise TypeError("Sampler is none, you should train pixelcnn model first")
            return 
        
        priors = np.zeros( (num_samples,  ) +self.pixel_cnn.input_shape, dtype = np.int32)

        rows, cols = self.pixel_cnn.input_shape

        for row in range(rows) : 
            for col in range(cols):
                probs= self.sampler.pridect(priors)
                priors[:,row,col] = probs[:,row, col]

        # retrieve quantized vectors correspending to each integer in priors

        quantized = self.vq_layer.get_quantized(priors)
        return self.decode(quantized)
         



    def generate(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        return self(x)[0]