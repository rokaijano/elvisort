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

#tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) 
#tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[1], True)


input_dir = "data\\generated_data\\fiath\\ARat_2017_05_31__0_002.dat"
model_dir = "trained_models\\fiath_clustering\\ARat_2017_05_31__0_002.dat"

unique_config_dir = os.path.join(input_dir,  "data_config.xml")

config={}
config["epochs"] = 200
config["batch_size"] = 8192
config["load_pretrained"] = True
config["fine_tune_detector"] = True
config["losses"] = {"alpha": 1024,"ce_eq":1, "beta":15,"delta":1, "theta":0}
config["beta_scheduler_enabled"] = False
config["only_positives"] = False

config["clustering_mode"] = False
config["cluster_update_interval"] = 5
config["gpu"] = 1

data = DatasetLoader(data_config_file=unique_config_dir, 
	batch_size=config["batch_size"], 
	only_positives=config["only_positives"], 
	autoload=False, 
	for_evaluation=True)



"""

"""
config["load_pretrained"] = True
config["fine_tune_detector"] = True
config["max_clusters"] = data.max_clusters
config["use_multi_gpu"] = False
config["timespan"] = data.image_shape[0]
config["pretrained_model_path"] = model_dir

config["exclude_decoder"] = True


vae_model = VAE_sorter(config=config)
vae_model.build()

data.load_test_dataset()

import time 
import csv


recording_freq = 20000

def mean_test_timing():
	all_times = []
	all_runtime = []
	for i in range(2): 
		t1 = time.time()
		detection, _, latents, gt = vae_model.vae.predict(data.test_dataset)

		runtime= time.time()-t1
		all_runtime.append(runtime)
		all_times.append((latents.shape[0]*vae_model.config["image_shape"][0])/(recording_freq*runtime))


	mean_time = sum(all_times) / len(all_times)
	mean_runtime = sum(all_runtime) / len(all_runtime)
	print("Mean time for the test to run: ", mean_time)
	print("Mean runtime for the test to run: ", mean_runtime)



def timing_delta_per_steps():
	all_times = []
	all_runtime = []
	for step in range(1,500,50): 
		t1 = time.time()
		detection, _, latents, gt = vae_model.vae.predict(data.test_dataset, steps= step)

		runtime= time.time()-t1

		all_runtime.append(runtime)
		all_times.append((latents.shape[0]*vae_model.config["image_shape"][0])/(recording_freq*runtime))

	with open("model_speed_measurements.csv", "w", newline="") as fd:
		writer = csv.writer(fd)
		writer.writerow(list(range(1,500,50)))
		writer.writerow(all_times)
		writer.writerow(all_runtime)

	mean_time = sum(all_times) / len(all_times)
	mean_runtime = sum(all_runtime) / len(all_runtime)
	print("Mean time for the test to run: ", mean_time)
	print("Mean runtime for the test to run: ", mean_runtime)


#mean_test_timing()
timing_delta_per_steps()

