

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

data_dir = "data\\generated_data\\spikeforest\\hybrid_janelia"
model_dir = "trained_models\\spikeforest"

config = {}
config["load_pretrained"] = True
config["fine_tune_detector"] = True

data = DatasetLoader(data_config_file=os.path.join(data_dir, "data_config.xml"))


config["max_clusters"] = data.max_clusters
config["pretrained_model_path"] = model_dir

vae_model = VAE_sorter(config_file=os.path.join(data_dir, "model_config.xml"), config=config)
vae_model.build()

vae_model.calcValTestRP(data.val_dataset, data.test_dataset, target_dir=model_dir)
