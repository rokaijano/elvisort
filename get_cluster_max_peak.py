import os
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import tensorflow as tf 
import csv

import time 

from tensorflow import TensorShape

from DatasetLoader import *
from models.vae_model import *
import csv

def custom_parse_fn(self, proto):

	features = {'image': tf.io.FixedLenFeature([], tf.string), 'label': tf.io.FixedLenFeature([], tf.string)}
	parsed_features = tf.io.parse_single_example(proto, features)
	images = tf.io.decode_raw(parsed_features['image'], tf.int16)
	labels = tf.io.decode_raw(parsed_features['label'], tf.int32)
	labels = tf.reshape(labels, (self.max_clusters,))
	images = tf.reshape(images, self.image_shape)
	images = tf.cast(images, tf.float32)

	labels = tf.cast(labels, tf.float32)

	if tf.math.reduce_sum(labels) < 2:
		label_with_max = labels * (tf.math.reduce_max(tf.math.abs(images)))
	else: 
		label_with_max = tf.zeros((self.max_clusters, ))

	images = (images / tf.math.reduce_max(images))
	return (images,label_with_max), (images, labels, labels, [0], [0])

DatasetLoader.parse_fn = custom_parse_fn

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) 
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[1], True)


input_dir = "data\\generated_data\\spikeforest\\hybrid_janelia"
model_dir = "trained_models\\spikeforest\\hybrid_janelia"

unique_config_dir = os.path.join(input_dir,  "data_config.xml")

config={}
config["epochs"] = 200
config["batch_size"] = 8192
config["load_pretrained"] = True
config["fine_tune_detector"] = True
config["losses"] = {"alpha": 1024,"ce_eq":1, "beta":15,"delta":1, "theta":0}
config["beta_scheduler_enabled"] = False
config["only_positives"] = False
config["exclude_decoder"] = False

config["clustering_mode"] = False
config["cluster_update_interval"] = 5
config["gpu"] = 1

data = DatasetLoader(data_config_file=unique_config_dir, 
	batch_size=config["batch_size"], 
	only_positives=config["only_positives"], 
	autoload=False, 
	for_evaluation=config["exclude_decoder"])


"""

"""
config["load_pretrained"] = True
config["fine_tune_detector"] = True
config["max_clusters"] = data.max_clusters
config["multi_label"] = data.data_config["multi_label"]
config["use_multi_gpu"] = False
config["timespan"] = data.image_shape[0]
config["pretrained_model_path"] = model_dir




vae_model = VAE_sorter(config=config)
vae_model.build()

data.load_test_dataset()
 

_, detection, _, latents, gt_max_peak = vae_model.vae.predict(data.test_dataset)

with open("cluster_mean_peaks.csv", "w", newline='') as fd:
	writer = csv.writer(fd)

	writer.writerow(["Mean uV", "Max uV", "Min uV"])
	for n_cluster in range(gt_max_peak.shape[1]):
		clust = [x[n_cluster] for x in gt_max_peak if x[n_cluster] > 0]

		if len(clust) == 0:
			writer.writerow([-1,-1,-1])
			continue

		clust = np.asarray(clust)
		writer.writerow([np.mean(clust), np.max(clust), np.min(clust)])

