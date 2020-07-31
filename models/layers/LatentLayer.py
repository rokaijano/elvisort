from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from models.layers.KLDivergenceLayer import *

class LatentLayer(Layer):

    def __init__(self, beta = 0.2, **kwargs):
        super(LatentLayer, self).__init__(**kwargs)
        self.beta = 0.2

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(LatentLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        z_mean, z_log_var = x
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        
        epsilon = K.random_normal(shape=(batch, dim))


        z_mean, z_log_var = KLDivergenceLayer(beta=self.beta)([z_mean, z_log_var])

        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return shape_a

    def get_config(self):
        return {}

    def from_config(cls, config):
        return cls()
    def from_config(config):
        return LatentLayer()



