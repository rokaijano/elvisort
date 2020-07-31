#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import os
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import tensorflow as tf 
import csv

import time 

from tensorflow import TensorShape

from DatasetLoader import *
from models.vae_model import *

## Experimental

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) 
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[1], True)


res_list = ["Log name", "Reconstruction loss", "Detection loss", "Val R", "Val P", "Test R", "Test P"]
#with open('fiath_all_results.csv', 'a') as fd:
#    writer = csv.writer(fd)
#    writer.writerow(res_list)


input_dir = "data\\generated_data\\spikeforest\\hybrid_janelia"
model_dir = "trained_models\\spikeforest\\hybrid_janelia"

unique_config_dir = os.path.join(input_dir,  "data_config.xml")
model_config_dir  = os.path.join(model_dir,  "model_config.xml")

config={}
config["epochs"] = 200
config["batch_size"] = 128
config["load_pretrained"] = False
config["fine_tune_detector"] = False
config["losses"] = {"alpha": 1024,"ce_eq":1, "beta":15,"delta":1, "theta":0}
config["beta_scheduler_enabled"] = False
config["only_positives"] = False

config["clustering_mode"] = False
config["cluster_update_interval"] = 5
config["gpu"] = 1

data = DatasetLoader(data_config_file=unique_config_dir, batch_size=config["batch_size"], only_positives=config["only_positives"], autoload=False)

#data.load_train_dataset()
data.load_val_dataset()

"""

"""
config["load_pretrained"] = True
config["fine_tune_detector"] = True
config["max_clusters"] = data.max_clusters
config["use_multi_gpu"] = False
config["multi_label"] = data.data_config["multi_label"]
config["timespan"] = data.image_shape[0]
config["pretrained_model_path"] = model_dir


vae_model = VAE_sorter(config_file=model_config_dir, config=config)
vae_model.build()
#hist = vae_model.fit(data.train_dataset, data.val_dataset)

#time.sleep(1)
#vae_model.save_weights_separate(custom_dir=model_dir, save_vae=False, save_detector=False)

data.load_test_dataset()
vae_model.calcValTestRP(data.val_dataset, data.test_dataset, target_dir = model_dir)


