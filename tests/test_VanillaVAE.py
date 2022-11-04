import tensorflow as tf
import unittest
from models.VanillaVAE import *
import warnings
import numpy as np 

class TestVAE(unittest.TestCase):

    def setUp(self) -> None:

        self.model = VanillaVAE(input_shape=(28,28,1),
                                hidden_dims=[16,32],
                                latent_dim=8)

    def test_call(self):
        x = tf.random.normal([16, 28, 28, 1])
        y = self.model(x)

        print(tf.shape(y[0]))

        self.assertTrue( (tf.shape(y[0]) == [16, 28, 28, 1]).numpy().all())
        self.assertTrue( (tf.shape(y[1]) == [16,8]).numpy().all())
        self.assertTrue( (tf.shape(y[2]) == [16,8]).numpy().all())


    def test_summary(self):
        
        x = tf.random.normal((16, 28, 28, 1))
        self.model(x)

        print(self.model.summary())



    def test_loss(self):
        x = tf.random.normal((16, 28, 28, 1))

        loss = self.model.loss_function(x)
        #self.assertTrue( np.testing.assert_almost_equal(loss[0].numpy(), loss[1].numpy() + loss[2].numpy(), decimal =2)) 
        print(loss[0])
        print(loss[1])
        print(loss[2])


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    unittest.main()