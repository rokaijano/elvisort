from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf 
import csv

from tensorflow import TensorShape

from DatasetLoader import *
from models.vae_model import *
import copy

loader = True

dataloader = DatasetLoader()

losses={"alpha":512,"beta":15,"delta":5,"theta":1, "kappa":0, "ce_eq":1}
config = {}
config.setdefault("losses", losses)
#config["training_steps"] = data.training_steps
config.setdefault("test_on_batch", False)
config.setdefault("clustering_mode", True)
config.setdefault("binary_detection", False)
config.setdefault("meta_learning_data", False)
config.setdefault("use_latentloss", False)
config.setdefault("only_positives", False)

config.setdefault("epochs", 10)
config.setdefault("batch_size", 256)


config.setdefault("max_clusters", dataloader.max_clusters)

config.setdefault("load_pretrained", loader)
config.setdefault("fine_tune_detector", loader)
config.setdefault("freeze_ae", False)
config.setdefault("pretrained_model_path", "trained_models")
config.setdefault("global_detector_dropout", 0.5)
config.setdefault("global_ae_dropout", 0.5)
config.setdefault("initial_clustering_weight", 0.001)

train_data, train_labels = dataloader.get_numpy(dataloader.test_dataset, return_in_batches=False)

#train_data = np.reshape(train_data, ((-1,)+tuple(train_data.shape[2:])))
#train_labels = np.reshape(train_labels, (-1, train_labels.shape[-1]))

clusters = train_labels.shape[-1]
int_labels = np.argmax(train_labels, axis=1)

model = VAE_sorter(config_file="model_config.xml", config = config)
model.build(load_weights = config["load_pretrained"], freeze_ae=config["freeze_ae"], use_clustering=config["clustering_mode"])

def cluster_labels(prod=1):
	latent = model.encoder.predict([train_data, train_labels])
	#reconstruction = model.decoder.predict(latent)
	reconstruction = copy.deepcopy(train_data)
	# At this point we should have a one hot encoded label array so we have to reverse it to int
	return reconstruction, latent

	clustering_imgs = []
	clustering_latents = []

	for i in range(clusters):
		
		if len(train_data[int_labels == i]) > 0:		
			#latent[int_labels==i] = latent[int_labels==i] - np.nan_to_num(prod*config["initial_clustering_weight"]*(latent[int_labels==i] - np.nanmean(latent[int_labels==i], axis=0)))
			reconstruction[int_labels == i] = train_data[int_labels == i] - np.nan_to_num(prod*config["initial_clustering_weight"]*(train_data[int_labels == i] - np.nanmean(train_data[int_labels == i], axis=0)))
		

	latent = np.nan_to_num(latent)
	reconstruction = np.nan_to_num(reconstruction)
	return reconstruction, latent


if config["only_positives"]:
	train_data = train_data[int_labels > 0]
	train_labels = train_labels[int_labels > 0]
	int_labels = int_labels[int_labels > 0]


def batch_generator(data, labels, latent, batch_size = 32):
    indices = np.arange(len(data)) 
    batch=[]
    while True:
            np.random.shuffle(indices) 
            for i in indices:
                batch.append(i)
                if len(batch)==batch_size:
                	yield [data[batch], labels[batch]], [data[batch], labels[batch],labels[batch],latent[batch]]
                	batch=[]

use_generator = False

for i in range(0,20):
	reconstruction, latent = cluster_labels(i+1)
	if use_generator:
		train_generator = batch_generator(train_data,  train_labels, latent, batch_size=config["batch_size"])	
		model.vae.fit(train_generator, steps_per_epoch=len(train_data)//config["batch_size"], epochs=config["epochs"])
	else:
		model.vae.fit([train_data, train_labels], [reconstruction, train_labels, train_labels, latent], batch_size=config["batch_size"], epochs=config["epochs"], validation_split=0.2)

	model.save_weights_separate()



	#val_data, val_labels = dataloader.get_numpy(dataloader.val_dataset, return_in_batches=False)
	#model.vae.eva
	#exit()
	
	