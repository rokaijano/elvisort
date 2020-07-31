
import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import backend as K
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class ClusteringLayer(Layer):

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):

        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
            
        return q  

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class UpdateClusters(tf.keras.callbacks.Callback):

    def __init__(self, data, parent_class):
        self.data = data 
        self.max_clusters = parent_class.config["max_clusters"]
        self.use_clustering = parent_class.config["clustering_mode"]
        self.cluster_update_interval =parent_class.config["cluster_update_interval"]
        self.vae = parent_class.vae
    def on_epoch_end(self, epoch, logs=None):
        
        if not self.use_clustering:
            return
        
        if epoch % self.cluster_update_interval != 0:
            return
            
        kmeans = KMeans(n_clusters=self.max_clusters, n_init=10)
        
        _, _, _, latent,_ = self.vae.predict(self.data, steps=100)
        
        latent = tf.where(tf.math.is_nan(latent), tf.zeros_like(latent), latent)
        latent = tf.where(tf.math.is_inf(latent), tf.zeros_like(latent), latent)

        kmeans.fit(latent)
        
        self.vae.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
         
