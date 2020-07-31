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
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


res_list = ["Log name", "Reconstruction loss", "Detection loss", "Val R", "Val P", "Test R", "Test P"]
#with open('fiath_all_results.csv', 'a') as fd:
#    writer = csv.writer(fd)
#    writer.writerow(res_list)

file_list = [
	
	#"ARat_2016_07_18__0_002.dat", 
    #"ARat_2016_07_20__0_002.dat",
    #"ARat_2016_07_27__0_002.dat",
    #"ARat_2016_10_12__0_002.dat",
    #"ARat_2016_10_19__0_002.dat",
    #"ARat_2017_04_05__0_002.dat",
    #"ARat_2017_05_19__0_002.dat",
    "ARat_2017_05_31__0_002.dat",  
    #"ARat_2017_06_01__0_002.dat"
]
input_dir = "data\\generated_data\\fiath\\"
output_dir = "trained_models\\fiath\\"

for file in file_list:

	unique_config_dir = os.path.join(input_dir, file, "data_config.xml")
	save_dir = os.path.join(output_dir, file)
	Path(save_dir).mkdir(parents=True, exist_ok=True)

	config={}
	config["epochs"] = 75
	config["batch_size"] = 256
	config["load_pretrained"] = False
	config["clustering_mode"] = True
	config["fine_tune_detector"] = False
	config["losses"] = {"ce_eq":1, "beta":15, "theta":10}
	config["beta_scheduler_enabled"] = True

	print(unique_config_dir)
	data = DatasetLoader(data_config_file=unique_config_dir, batch_size=config["batch_size"], only_binary_labels=True, autoload=False)

	data.load_train_dataset()
	data.load_val_dataset()

	config["max_clusters"] = 2
	config["use_multi_gpu"] = False

	vae_model = VAE_sorter(config_file="model_config.xml", config=config)
	vae_model.build()
	hist = vae_model.fit(data.train_dataset, data.val_dataset)

	time.sleep(1)
	vae_model.save_weights_separate(custom_dir=save_dir)

	data.load_test_dataset()
	vae_model.calcValTestRP(data.val_dataset, data.test_dataset, target_dir = save_dir)

