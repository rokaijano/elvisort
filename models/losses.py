from tensorflow.keras.losses import mse, binary_crossentropy, categorical_crossentropy, kullback_leibler_divergence
from tensorflow.keras import backend as K
import tensorflow as tf

def vae_loss(valid_loss = True, alpha=4096):
    def loss(original, reconstructed):
        
        if not valid_loss:
            return 0.

        reconstruction_loss = mse(K.flatten(original), K.flatten(reconstructed))
        reconstruction_loss *= alpha

        return K.mean(reconstruction_loss)

    return loss

def latent_loss(valid_loss = True, kappa=1):
    def loss(original, reconstructed):
        
        if not valid_loss:
            return 0.

        reconstruction_loss = mse(K.flatten(original), K.flatten(reconstructed))

        return K.mean(reconstruction_loss) * kappa

    return loss

def custom_crossentropy(use_delta=True, valid_loss = True, delta=10, ce_eq=1):
    def loss(y_true, y_pred):
        if not valid_loss:
            return 0.
        cce = categorical_crossentropy(y_true, y_pred)

        return delta*cce

        loss_ = ce_eq * (1 - y_true[:, 0]) * cce + y_true[:, 0] * cce
        
        if use_delta:
            loss_ = delta * loss_
            
        return loss_

    return loss

def zero_loss(y_true, y_pred):
    return 0.

def cluster_loss(use_theta = True, valid_loss = True, theta=1):
    
    def loss(y_true, layer_output):
        
        if not valid_loss:
            return 0.
        
        if use_theta:
            layer_output = theta * kullback_leibler_divergence(y_true, layer_output)
        
        return layer_output
    
    return loss
        
