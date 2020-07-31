from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class KLDivergenceLayer(Layer):

    def __init__(self, beta=0.2, **kwargs):
        self.is_placeholder = True
        self.beta = beta
        super(KLDivergenceLayer, self).__init__(**kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)

        self.add_loss(self.beta * K.mean(kl_batch), inputs=inputs)

        return inputs

    def get_config(self):
        return beta

    def from_config(cls, config):
        return cls(beta=config)
