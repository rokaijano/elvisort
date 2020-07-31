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

file_list = [
	
	""
]
input_dir = "data\\generated_data\\spikeforest\\hybrid_janelia"
output_dir = "trained_models\\spikeforest\\hybrid_janelia"

for file in file_list:
	try:
		unique_config_dir = os.path.join(input_dir, file, "data_config.xml")
		save_dir = os.path.join(output_dir, file)
		Path(save_dir).mkdir(parents=True, exist_ok=True)

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

		print(unique_config_dir)
		data = DatasetLoader(data_config_file=unique_config_dir, batch_size=config["batch_size"], only_positives=config["only_positives"], autoload=False)

		data.load_train_dataset()
		data.load_val_dataset()

		config["max_clusters"] = data.max_clusters
		config["use_multi_gpu"] = False
		config["multi_label"] = data.data_config["multi_label"]
		config["timespan"] = data.image_shape[0]
		

		vae_model = VAE_sorter(config_file="model_config.xml", config=config)
		vae_model.build()
		hist = vae_model.fit(data.train_dataset, data.val_dataset)

		time.sleep(1)
		vae_model.save_weights_separate(custom_dir=save_dir)

		data.load_test_dataset()
		vae_model.calcValTestRP(data.val_dataset, data.test_dataset, target_dir = save_dir)
	except Exception as e:
		print(e)
		continue

